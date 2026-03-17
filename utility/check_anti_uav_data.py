import pandas as pd
import os

data_dir = '/cluster/scratch/nbaruffol/anti_uav_formatted'
csv_path = f"{data_dir}/part1/ImageSets/groundtruth.csv"

df = pd.read_csv(csv_path)
first_row = df.iloc[0]
flight_id = first_row['flight_id']

# Replicate dataset_tracking.py logic
base_img_name = first_row['img_name'][:-4] 
expected_rgb = f"{data_dir}/part1/Images/{flight_id}/{base_img_name}_rgb.jpg"
expected_ir = f"{data_dir}/part1/Images/{flight_id}/{base_img_name}_ir.jpg"
expected_transform = f"{data_dir}/frame_transforms/part1/{flight_id}.pkl"

print(f"Expects RGB: {expected_rgb} -> Exists? {os.path.exists(expected_rgb)}")
print(f"Expects IR:  {expected_ir} -> Exists? {os.path.exists(expected_ir)}")
print(f"Expects PKL: {expected_transform} -> Exists? {os.path.exists(expected_transform)}")
