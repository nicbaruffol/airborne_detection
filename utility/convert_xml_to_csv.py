import xml.etree.ElementTree as ET
import pandas as pd
import math
import cv2
import os

def extract_frames(video_path, output_folder, prefix="frame_"):
    # Create the folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        # Save each frame as a PNG to match the format the dataset expects
        file_path = os.path.join(output_folder, f"{prefix}{count:04d}.png")
        cv2.imwrite(file_path, frame)
        count += 1
        
    cap.release()
    print(f"Done! Extracted {count} frames to {output_folder}")


def convert_cvat_xml_to_csv(xml_file, output_csv, flight_id="my_custom_flight"):
    # Load the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []

    # CVAT Video format uses <track> for objects and <box> for frames
    for track in root.findall('track'):
        track_id = track.get('id')
        label = track.get('label')
        
        # The dataset script expects the ID to be the class name + a number (e.g., "Drone1")
        obj_id = f"{label}{track_id}"

        for box in track.findall('box'):
            # If the object left the screen, CVAT marks outside="1". We skip those.
            if box.get('outside') == '1':
                continue 

            frame_num = int(box.get('frame'))
            
            # Pad the frame number to match your image filenames (e.g., frame_0000.png)
            # Change this if your images are named differently!
            img_name = f"frame_{frame_num:04d}" 

            # Extract coordinates
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # Fill in required challenge columns with safe defaults
            # (Unless you specifically annotated distance, we put a dummy value)
            distance = 100.0 
            is_above_horizon = 1.0 # Defaulting to flying in the sky

            data.append({
                'flight_id': flight_id,
                'img_name': img_name,
                'frame': frame_num,
                'range_distance_m': distance,
                'gt_left': xtl,
                'gt_right': xbr,
                'gt_top': ytl,
                'gt_bottom': ybr,
                'id': obj_id,
                'is_above_horizon': is_above_horizon
            })

    # Create a DataFrame and save it
    df = pd.DataFrame(data)
    
    # Sort chronologically
    if not df.empty:
        df = df.sort_values('frame')
        
    df.to_csv(output_csv, index=False)
    print(f"Successfully converted {len(df)} bounding boxes to {output_csv}!")

if __name__ == "__main__":
    # Run the conversion
    convert_cvat_xml_to_csv(
        xml_file="/cluster/scratch/nbaruffol/raw_videos/annotations.xml", 
        output_csv="/cluster/scratch/nbaruffol/raw_videos/groundtruth.csv", 
        flight_id="my_custom_flight" # Make sure this matches your folder name!
    )
    # # Run it on your video
    # extract_frames(
    #     video_path="/cluster/scratch/nbaruffol/raw_videos/rgb_fixed_1.mp4", 
    #     output_folder="/cluster/scratch/nbaruffol/raw_videos/custom_dataset/part1/Images/my_custom_flight"
    # )
