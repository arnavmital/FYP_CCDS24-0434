import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.exceptions import NotFittedError
from causalai.models.time_series.pc import PC
from causalai.data.time_series import TimeSeriesData
from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.data.transforms.time_series import StandardizeTransform

input_dir = "/home/mittal1/DriveTester/scenario_csv"
output_dir = "/home/mittal1/DriveTester/causal_graphs"
os.makedirs(output_dir, exist_ok=True)

csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
print(f"🧾 Found {len(csv_files)} scenario CSVs.")

for fname in csv_files:
    try:
        fpath = os.path.join(input_dir, fname)
        df = pd.read_csv(fpath)

        # Filter: skip files with too few rows or no collisions
        if df.shape[0] < 20 or 'collision_label' not in df.columns or df['collision_label'].sum() == 0:
            print(f"⏭️ Skipping {fname}: too few rows or no collisions.")
            continue

        # Extract and preserve collision_label
        collision_col = df['collision_label']
        df = df.select_dtypes(include=[np.number])
        df = df.loc[:, df.std() > 1e-6]
        if 'collision_label' not in df.columns:
            df['collision_label'] = collision_col

        # Standardize
        features_to_standardize = [col for col in df.columns if col != 'collision_label']
        transform = StandardizeTransform()
        transform.fit(df[features_to_standardize].values)
        data_trans = transform.transform(df[features_to_standardize].values)

        # Recombine with label
        data_trans = np.concatenate([data_trans, df[['collision_label']].values], axis=1)
        var_names = features_to_standardize + ['collision_label']
        ts_data = TimeSeriesData(data_trans, var_names=var_names)

        # PC algorithm
        ci_test = PartialCorrelation()
        pc = PC(data=ts_data, CI_test=ci_test, max_lag=1)
        result = pc.run(pvalue_thres=0.01, max_condition_set_size=2)

        # Extract parents of collision_label
        graph = {var: result[var]['parents'] for var in result}
        collision_parents = graph.get('collision_label', [])
        if not collision_parents:
            print(f"⚠️ No parents for collision_label in {fname}")
            continue

        # Draw and save graph with colored nodes
        G = nx.DiGraph()
        G.add_node("collision_label", color='lightcoral')

        for parent in collision_parents:
            parent_var = parent[0]
            if parent_var != 'collision_label':
                G.add_node(parent_var, color='lightblue')
                G.add_edge(parent_var, 'collision_label')

        pos = nx.spring_layout(G, seed=42)
        node_colors = [G.nodes[n].get('color', 'lightgrey') for n in G.nodes()]

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2500,
                font_size=10, arrowsize=20)
        plt.title(f"Causal Parents of Collision\n{fname}")
        out_path = os.path.join(output_dir, fname.replace(".csv", "_causal_graph.png"))
        plt.savefig(out_path)
        plt.close()
        print(f"✅ Saved graph for {fname}")

    except Exception as e:
        print(f"❌ Failed {fname}: {e}")