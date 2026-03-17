import os
import cv2
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_rgb_transforms():
    data_dir = '/cluster/scratch/nbaruffol/anti_uav_formatted'
    csv_path = f"{data_dir}/part1/ImageSets/groundtruth.csv"
    transforms_dir = f"{data_dir}/frame_transforms/part1"
    os.makedirs(transforms_dir, exist_ok=True)

    print("Loading ground truth CSV...")
    df = pd.read_csv(csv_path)
    flights = df['flight_id'].unique()

    for flight_id in tqdm(flights, desc="Generating RGB Transforms"):
        # Sort by frame to ensure consecutive ordering
        flight_df = df[df['flight_id'] == flight_id].sort_values('frame')
        
        transforms = {}
        prev_gray = None
        
        for _, row in flight_df.iterrows():
            img_name = row['img_name']
            base_img_name = img_name[:-4] 
            rgb_path = f"{data_dir}/part1/Images/{flight_id}/{base_img_name}_rgb.jpg"
            
            if not os.path.exists(rgb_path):
                continue
                
            # Read ONLY the RGB image in grayscale for motion estimation
            curr_gray = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
            
            if prev_gray is None:
                # First frame has 0 offset
                transforms[img_name] = [0.0, 0.0]
            else:
                # Calculate translation using Phase Correlation
                shift, _ = cv2.phaseCorrelate(np.float32(prev_gray), np.float32(curr_gray))
                transforms[img_name] = [float(shift[0]), float(shift[1])]
                
            prev_gray = curr_gray
            
        # Save the transformations to the PKL file expected by train.py
        pkl_path = os.path.join(transforms_dir, f"{flight_id}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(transforms, f)

if __name__ == "__main__":
    compute_rgb_transforms()
    print("Successfully generated all RGB transforms!")