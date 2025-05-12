import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from causalai.models.time_series.pc import PC
from causalai.data.time_series import TimeSeriesData
from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.models.time_series.causal_inference import CausalInference
from causalai.data.transforms.time_series import StandardizeTransform

input_dir = "/home/mittal1/DriveTester/scenario_csv"
output_dir = "/home/mittal1/DriveTester/ate_graphs"
long_summary_path = "ate_summary_all.csv"
left_summary_path = "ate_summary_left_turn.csv"
right_summary_path = "ate_summary_right_turn.csv"
straight_summary_path = "ate_summary_straight.csv"

os.makedirs(output_dir, exist_ok=True)

# Define maneuver categories
left_turn_scenarios = {"lane31_lane9", "lane23_lane27"}
straight_scenarios = {"lane31_lane15", "lane0_lane7", "lane23_lane24", "lane20_lane27"}
right_turn_scenarios = {"lane0_lane14", "lane23_lane21"}

def get_maneuver_type(fname):
    for key in left_turn_scenarios:
        if key in fname: return "left_turn"
    for key in right_turn_scenarios:
        if key in fname: return "right_turn"
    for key in straight_scenarios:
        if key in fname: return "straight"
    return "unknown"

csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
ate_records = []

for fname in tqdm(csv_files, desc="📊 Recomputing ATEs"):
    try:
        fpath = os.path.join(input_dir, fname)
        df = pd.read_csv(fpath)

        if 'collision_label' not in df.columns or df['collision_label'].sum() == 0 or df.shape[0] < 20:
            continue

        maneuver_type = get_maneuver_type(fname)

        collision_col = df['collision_label']
        df = df.select_dtypes(include=[np.number])
        df = df.loc[:, df.std() > 1e-6]
        if 'collision_label' not in df.columns:
            df['collision_label'] = collision_col

        features_to_standardize = [col for col in df.columns if col != 'collision_label']
        transform = StandardizeTransform()
        transform.fit(df[features_to_standardize].values)
        data_trans = transform.transform(df[features_to_standardize].values)
        data_trans = np.concatenate([data_trans, df[['collision_label']].values], axis=1)
        var_names = features_to_standardize + ['collision_label']

        ts_data = TimeSeriesData(data_trans, var_names=var_names)

        pc = PC(data=ts_data, CI_test=PartialCorrelation(), max_lag=1)
        result = pc.run(pvalue_thres=0.01, max_condition_set_size=2)

        causal_graph = {var: result[var]['parents'] for var in result}
        collision_parents = causal_graph.get('collision_label', [])
        parent_vars = [p[0] for p in collision_parents if p[0] != 'collision_label']

        if not parent_vars:
            continue

        inference = CausalInference(
            data=data_trans,
            var_names=var_names,
            causal_graph=causal_graph,
            prediction_model=LinearRegression,
            use_multiprocessing=False,
            discrete=False
        )

        G = nx.DiGraph()
        G.add_node("collision_label", color='lightcoral')

        for var in parent_vars:
            if var not in df.columns:
                continue
            values = df[var].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            treatment_val = mean_val + std_val
            control_val = mean_val - std_val

            treatment = {
                'var_name': var,
                'treatment_value': treatment_val * np.ones_like(values),
                'control_value': control_val * np.ones_like(values)
            }


            ate, _, _ = inference.ate('collision_label', treatment)
            ate_records.append({
                "File": fname,
                "Feature": var,
                "ATE": ate,
                "Num_Collision_Frames": int(collision_col.sum()),
                "Maneuver": maneuver_type
            })

            G.add_node(var, color='lightblue')
            G.add_edge(var, 'collision_label', weight=ate)

        # Save ATE graph
        pos = nx.spring_layout(G, seed=42)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.4f}" for u, v in G.edges()}
        node_colors = [G.nodes[n].get('color', 'lightgrey') for n in G.nodes()]
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2500, font_size=10, arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=9)
        plt.title(f"{fname} - ATE Causal Graph")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname.replace(".csv", "_ate_graph.png")))
        plt.close()

    except Exception as e:
        print(f"❌ Error in {fname}: {e}")

# === Save CSVs ===
ate_df = pd.DataFrame(ate_records)
ate_df.to_csv(long_summary_path, index=False)
print(f"✅ Saved overall ATEs to {long_summary_path}")

# Save maneuver-specific CSVs
ate_df[ate_df['Maneuver'] == 'left_turn'].to_csv(left_summary_path, index=False)
ate_df[ate_df['Maneuver'] == 'right_turn'].to_csv(right_summary_path, index=False)
ate_df[ate_df['Maneuver'] == 'straight'].to_csv(straight_summary_path, index=False)
print("✅ Saved maneuver-specific ATE CSVs.")