import os
import json
import cv2
import pandas as pd
from tqdm import tqdm

# We will read from your raw download and write to a new clean folder
SRC_DIR = "/cluster/scratch/nbaruffol/anti_uav/train"
DST_DIR = "/cluster/scratch/nbaruffol/anti_uav_formatted/part1"

IMAGES_DIR = os.path.join(DST_DIR, "Images")
IMAGESETS_DIR = os.path.join(DST_DIR, "ImageSets")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(IMAGESETS_DIR, exist_ok=True)

csv_rows = []

# Get all flight folders
flights = [f for f in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, f))]

print("Extracting frames and building groundtruth.csv...")
for flight_id in tqdm(flights):
    flight_path = os.path.join(SRC_DIR, flight_id)
    
    # Paths to the raw Anti-UAV files
    vis_json_path = os.path.join(flight_path, "visible.json")
    vis_mp4 = os.path.join(flight_path, "visible.mp4")
    ir_mp4 = os.path.join(flight_path, "infrared.mp4")
    
    if not os.path.exists(vis_json_path) or not os.path.exists(vis_mp4) or not os.path.exists(ir_mp4):
        continue
        
    with open(vis_json_path, 'r') as f:
        vis_data = json.load(f)
        
    # Anti-UAV stores the bounding boxes in 'gt_rect' or 'position'
    boxes = vis_data.get('gt_rect', vis_data.get('position', []))
    exists = vis_data.get('exist', [])
    
    cap_vis = cv2.VideoCapture(vis_mp4)
    cap_ir = cv2.VideoCapture(ir_mp4)
    
    flight_out_dir = os.path.join(IMAGES_DIR, flight_id)
    os.makedirs(flight_out_dir, exist_ok=True)
    
    frame_idx = 0
    while True:
        ret_vis, frame_vis = cap_vis.read()
        ret_ir, frame_ir = cap_ir.read()
        
        if not ret_vis or not ret_ir:
            break
            
        img_name = f"frame_{frame_idx:04d}"
        
        # 1. Save frames as JPGs with the exact suffixes we put in dataset_tracking.py
        cv2.imwrite(os.path.join(flight_out_dir, f"{img_name}_rgb.jpg"), frame_vis)
        cv2.imwrite(os.path.join(flight_out_dir, f"{img_name}_ir.jpg"), frame_ir)
        
        # 2. Parse the bounding boxes if the drone is in the frame
        if frame_idx < len(exists) and frame_idx < len(boxes):
            if exists[frame_idx] == 1:
                x, y, w, h = boxes[frame_idx]
                if w > 0 and h > 0:
                    csv_rows.append({
                        'flight_id': flight_id,
                        # Your dataset_tracking.py explicitly strips the last 4 chars ([:-4]) 
                        # so we MUST include a dummy .jpg extension here
                        'img_name': f"{img_name}.jpg", 
                        'frame': frame_idx,
                        'range_distance_m': 100.0,      # Dummy distance to satisfy tracking bounds
                        'gt_left': x,
                        'gt_right': x + w,
                        'gt_top': y,
                        'gt_bottom': y + h,
                        'id': 'Drone1',                 # Your regex explicitly looks for Letters+Numbers
                        'is_above_horizon': 1           # Dummy value
                    })
        
        frame_idx += 1
        
    cap_vis.release()
    cap_ir.release()

# 3. Save the master CSV just like the Amazon challenge provided
df = pd.DataFrame(csv_rows)
df.to_csv(os.path.join(IMAGESETS_DIR, "groundtruth.csv"), index=False)
print(f"Success! Extracted {len(df)} annotated pairs into {DST_DIR}")