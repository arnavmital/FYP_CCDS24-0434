import os
import pickle
import pandas as pd
import numpy as np
from math import hypot
from tqdm import tqdm

# === CONFIG ===
COLLISION_THRESHOLD = 0.01
input_dir = "/home/mittal1/DriveTester/extracted_files"
output_dir = "/home/mittal1/DriveTester/scenario_csv"
summary_csv = "/home/mittal1/DriveTester/collision_extraction_summary_final.csv"
os.makedirs(output_dir, exist_ok=True)

def min_bbox_edge_distance(bbox1, loc1, bbox2, loc2):
    x1_min, x1_max = loc1['x'] - bbox1['width']/2, loc1['x'] + bbox1['width']/2
    y1_min, y1_max = loc1['y'] - bbox1['length']/2, loc1['y'] + bbox1['length']/2
    x2_min, x2_max = loc2['x'] - bbox2['width']/2, loc2['x'] + bbox2['width']/2
    y2_min, y2_max = loc2['y'] - bbox2['length']/2, loc2['y'] + bbox2['length']/2
    dx = max(x1_min - x2_max, x2_min - x1_max, 0)
    dy = max(y1_min - y2_max, y2_min - y1_max, 0)
    return np.hypot(dx, dy)

def extract_features_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    rows = []
    for frame_id in sorted(data.keys()):
        frame = data[frame_id]
        timestamp = frame['timestamp']
        actors = frame['actor_state']

        collided_ids = set()
        actor_ids = list(actors.keys())
        for i in range(len(actor_ids)):
            for j in range(i + 1, len(actor_ids)):
                a1 = actors[actor_ids[i]]
                a2 = actors[actor_ids[j]]
                d = min_bbox_edge_distance(a1['bbox'], a1['location'], a2['bbox'], a2['location'])
                if d < COLLISION_THRESHOLD:
                    collided_ids.update([actor_ids[i], actor_ids[j]])

        ego = next((v for v in actors.values() if v['role'] in ['ego', 'apollo', 'hero'] or v['id'] == 0), None)
        if not ego:
            continue

        ego_speed = ego['speed']
        ego_yaw = ego['location']['yaw']
        ego_loc = ego['location']
        ego_ctrl = ego.get('control', {})

        for actor_id, actor in actors.items():
            role = actor['role']
            loc = actor['location']
            ctrl = actor.get('control', {})
            dist = np.linalg.norm([loc['x'] - ego_loc['x'], loc['y'] - ego_loc['y']])
            rel_speed = actor['speed'] - ego_speed
            heading_diff = abs(ego_yaw - loc['yaw'])

            row = {
                'ego_speed': ego_speed,
                'ego_acceleration': ego['acceleration'],
                'ego_throttle': ego_ctrl.get('throttle', 0.0),
                'ego_brake': ego_ctrl.get('brake', 0.0),

                'npc_speed': actor['speed'] if role == 'vehicle' else 0.0,
                'npc_acceleration': actor['acceleration'] if role == 'vehicle' else 0.0,
                'npc_throttle': ctrl.get('throttle', 0.0) if role == 'vehicle' else 0.0,
                'npc_brake': ctrl.get('brake', 0.0) if role == 'vehicle' else 0.0,
                'npc_steering': ctrl.get('steering', 0.0) if role == 'vehicle' else 0.0,

                'walker_speed': actor['speed'] if role == 'walker' else 0.0,
                'walker_acceleration': actor['acceleration'] if role == 'walker' else 0.0,

                'static_object': 1 if role == 'static' else 0,

                'distance_to_ego': dist,
                'relative_speed_to_ego': rel_speed,
                'heading_diff_to_ego': heading_diff,
                'actor_yaw': loc['yaw'],

                'collision_label': int(actor_id in collided_ids),
            }
            rows.append(row)

    return pd.DataFrame(rows)

# === Batch processing ===
progress = []
pkl_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pkl")])

for fname in tqdm(pkl_files, desc="Processing scenarios"):
    try:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname.replace(".pkl", ".csv"))
        df = extract_features_from_pkl(in_path)
        df.to_csv(out_path, index=False)
        progress.append((fname, df.shape[0], df['collision_label'].sum()))
    except Exception as e:
        progress.append((fname, "ERROR", str(e)))


summary_df = pd.DataFrame(progress, columns=["File", "Rows", "Collisions"])
summary_df.to_csv(summary_csv, index=False)
print("✅ All scenarios processed. Summary saved to:", summary_csv)