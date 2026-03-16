import cv2
import os
import sys
from tqdm import tqdm
import numpy as np   # <--- Add this line!

# Add the tracker to the system path
current_path = os.getcwd()
sys.path.append(f'{current_path}/seg_tracker')

# Import your model's brain
from seg_tracker.seg_tracker import SegDetector, SegTrackerFromOffset

def track_custom_video(input_video_path, output_video_path):
    print("Loading AI Model...")
    # Initialize the detector and tracker exactly like seg_test.py does
    detector = SegDetector()
    tracker = SegTrackerFromOffset(detector=detector)

    # Open your video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}. Check the path!")
        return

    # Get video properties for saving the output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get original video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # The massive size the AI demands
    model_w, model_h = 2448, 2048

    # Calculate how much black padding goes on the left and top
    x_offset = (model_w - orig_width) // 2
    y_offset = (model_h - orig_height) // 2

    # We want to save the output video in YOUR original resolution (un-stretched!)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_width, orig_height))

    print(f"Processing {total_frames} frames from {input_video_path}...")
    
    prev_frame = None 
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 'frame' is 3-channel (even though it looks gray), so we can draw GREEN on it later.
        # We just need to extract a strict 1-channel array for the AI's math.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Pad the 1-channel grayscale frame to the massive 2448x2048 size
        padded_gray = np.zeros((model_h, model_w), dtype=np.uint8)
        padded_gray[y_offset:y_offset+orig_height, x_offset:x_offset+orig_width] = gray_frame

        # 3. THE MAGIC IR TRICK: Invert the pixels! (Dark IR sky -> Bright daytime sky)
        inverted_gray = cv2.bitwise_not(padded_gray)

        if prev_frame is None:
            prev_frame = inverted_gray.copy()

        # 4. Ask the AI to find the drones using the INVERTED tracking frames
        results = tracker.predict(image=inverted_gray, prev_image=prev_frame) 

        # 5. Draw the boxes on your ORIGINAL 3-channel 'frame'
        if results is not None:
            for res in results:
                if isinstance(res, dict):
                    cx, cy, w, h = res['cx'], res['cy'], res['w'], res['h']
                    track_id = res.get('track_id', -1)
                    conf = res.get('conf', 1.0)
                else: 
                    cx, cy, w, h = res.x, res.y, res.w, res.h
                    track_id = res.track_id
                    conf = res.confidence

                # Shift the AI's coordinates back to your video's 1440x1080 scale
                cx = cx - x_offset
                cy = cy - y_offset

                # Ignore hallucinations in the black border
                if cx < 0 or cx > orig_width or cy < 0 or cy > orig_height:
                    continue

                # Calculate the box corners
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)

                # Draw the green box directly on your original 3-channel video frame!
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID: {track_id} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 6. Save the native resolution frame with the green boxes
        out.write(frame)
        
        # Save the INVERTED gray canvas for the tracking motion math in the next loop
        prev_frame = inverted_gray.copy()


    # Clean up
    cap.release()
    out.release()
    print(f"Done! Saved tracked video to {output_video_path}")

if __name__ == "__main__":
    # --- CHANGE THESE PATHS ---
    INPUT_VIDEO = "/cluster/scratch/nbaruffol/raw_videos/rgb_fixed_1.mp4"    # Put your video file in the folder and put its name here
    OUTPUT_VIDEO = "/cluster/scratch/nbaruffol/raw_videos/rgb_tracked_1.mp4"   # The name of the file it will create
    
    track_custom_video(INPUT_VIDEO, OUTPUT_VIDEO)