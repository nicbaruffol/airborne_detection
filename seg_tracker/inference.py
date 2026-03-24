import torch
import cv2
import numpy as np
from tqdm import tqdm
import os
# Import your model builder and prediction decoder from your existing files
from train import build_model
import common_utils
import seg_prediction_to_items
import offset_grid_to_transform # Required for the author's math
from train_transformation import build_model as build_transform_model
import dataset_tracking
import matplotlib.pyplot as plt
import pandas as pd
import config


class DelayTracker:
    # We add max_coast_frames to let tracks survive short disappearances
    def __init__(self, min_steps=8, max_distance=40.0, max_coast_frames=3):
        self.min_steps = min_steps
        self.base_max_distance = max_distance
        self.max_coast_frames = max_coast_frames
        self.active_tracks = {}  
        self.next_track_id = 0

    def update(self, current_frame_detections):
        updated_tracks = {}
        unmatched_detections = []
        
        # Keep track of which active tracks got matched this frame
        matched_track_ids = set()

        # Step 1: Predict where current detections WERE in the previous frame
        for det in current_frame_detections:
            pred_prev_cx = det['cx'] - det['tracking'][0]
            pred_prev_cy = det['cy'] - det['tracking'][1]
            
            best_match_id = None
            best_dist = 999999.0

            # Step 2: Associate with existing active tracks
            for track_id, track in self.active_tracks.items():
                if track_id in matched_track_ids:
                    continue # Already matched to another detection

                dist = np.sqrt((pred_prev_cx - track['last_cx'])**2 + (pred_prev_cy - track['last_cy'])**2)
                
                # --- THE DYNAMIC VELOCITY TWEAK ---
                # Allow a larger error radius if the plane is moving fast
                dynamic_max_dist = self.base_max_distance
                if len(track['history']) > 2:
                    # Calculate speed over the last 2 frames
                    dx = track['history'][-1]['cx'] - track['history'][-2]['cx']
                    dy = track['history'][-1]['cy'] - track['history'][-2]['cy']
                    speed = np.sqrt(dx**2 + dy**2)
                    
                    # If it's moving fast (e.g., > 15 pixels per frame), expand the catch radius!
                    if speed > 15.0:
                        dynamic_max_dist = self.base_max_distance * 2.0 # Double the allowance

                if dist < dynamic_max_dist and dist < best_dist:
                    best_dist = dist
                    best_match_id = track_id

            if best_match_id is not None:
                # We found a match! Update the track
                track = self.active_tracks[best_match_id]
                track['age'] += 1
                track['missed_frames'] = 0 # Reset coasting counter!
                track['last_cx'] = det['cx']
                track['last_cy'] = det['cy']
                track['history'].append(det)
                
                updated_tracks[best_match_id] = track
                matched_track_ids.add(best_match_id)
            else:
                unmatched_detections.append(det)

        # Step 3: Handle Unmatched Tracks (THE COASTING TWEAK)
        for track_id, track in self.active_tracks.items():
            if track_id not in matched_track_ids:
                track['missed_frames'] += 1
                # Only keep it if it hasn't exceeded our coasting allowance
                if track['missed_frames'] <= self.max_coast_frames:
                    updated_tracks[track_id] = track

        # Step 4: Create new tracks for completely new detections
        for det in unmatched_detections:
            updated_tracks[self.next_track_id] = {
                'age': 1,
                'missed_frames': 0,
                'last_cx': det['cx'],
                'last_cy': det['cy'],
                'history': [det]
            }
            self.next_track_id += 1

        self.active_tracks = updated_tracks

        # Step 5: Only yield detections if the track is mature AND currently visible
        mature_detections = []
        for track_id, track in self.active_tracks.items():
            # If it's old enough, AND it was actually detected this frame (missed_frames == 0)
            if track['age'] >= self.min_steps and track['missed_frames'] == 0:
                mature_det = track['history'][-1]
                mature_det['track_id'] = track_id 
                mature_detections.append(mature_det)

        return mature_detections

def evaluate_tracker(experiment_name, tracking_weights, model_weights_path, transform_weights):
    # 2. LOAD YOUR TRAINED TRACKING MODEL
    cfg = common_utils.load_config_data(experiment_name)
    tracking_model = build_model(cfg)
    tracking_model = tracking_model.cuda()
    tracking_model.eval()
    
    # Load the weights exported by train.py
    checkpoint = torch.load(tracking_weights, weights_only=False)
    tracking_model.load_state_dict(checkpoint["model_state_dict"])

    # LOAD THE TSM TRANSFORMATION MODEL
    # Ensure you use the exact experiment name that matches your config file for '030_tr_tsn_rn34'
    experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")
    tr_cfg = common_utils.load_config_data("030_tr_tsn_rn34_w3_crop_borders", experiments_dir=experiments_dir)
    transform_model = build_transform_model(tr_cfg).cuda()
    
    # Load your trained weights for the transform model
    tr_checkpoint = torch.load(transform_weights, weights_only=False)
    transform_model.load_state_dict(tr_checkpoint["model_state_dict"])
    transform_model.eval()
    
    tracker = DelayTracker(min_steps=8, max_distance=40.0, max_coast_frames=3)

    # 2. LOAD VALIDATION DATASET
    # We use your existing TrackingDataset class to get the exact ground truth
    val_dataset = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_VALID,
        cfg_data=cfg,
        return_torch_tensors=False, # We want raw numpy arrays to process manually
        small_subset=False
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_frames = 0

    print("Running evaluation...")
    
    sample_frame = cv2.imread(
        val_dataset.img_fn(val_dataset.frames[0].part, val_dataset.frames[0].flight_id, val_dataset.frames[0].img_name),
        cv2.IMREAD_GRAYSCALE
    )
    vid_h, vid_w = sample_frame.shape

    fps = 30.0 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter("eval_output.mp4", fourcc, fps, (vid_w, vid_h), isColor=True)

    # We need to simulate the video stream by tracking flights
    # Group frames by flight_id so the tracker doesn't mix up different videos
    flights = {}
    for i in range(len(val_dataset)):
        frame_data = val_dataset.frames[i]
        if frame_data.flight_id not in flights:
            flights[frame_data.flight_id] = []
        flights[frame_data.flight_id].append(i)

    hit_widths = []
    miss_widths = []
    detection_confs_and_widths = []  

    for flight_id, frame_indices in tqdm(flights.items()):
        # Reset the tracker for each new video/flight!
        tracker = DelayTracker(min_steps=8, max_distance=40.0, max_coast_frames=3)
        prev_frame_aligned = None

        for idx in frame_indices:
            frame_data = val_dataset.frames[idx]
            current_frame_gray = cv2.imread(
                val_dataset.img_fn(frame_data.part, frame_data.flight_id, frame_data.img_name),
                cv2.IMREAD_GRAYSCALE
            )
            if current_frame_gray is None:
                continue

            if prev_frame_aligned is None:
                prev_frame_aligned = current_frame_gray
                continue

            orig_h, orig_w = current_frame_gray.shape

            # --- Frame Alignment + Network Prediction ---
            prev_frame_aligned = align_frames_dl(transform_model, current_frame_gray, prev_frame_aligned)
            tensor_inputs = prepare_inputs(current_frame_gray, prev_frame_aligned)
            with torch.cuda.amp.autocast(), torch.no_grad():
                pred = tracking_model(tensor_inputs)

            pred['mask'] = torch.sigmoid(pred['mask'])

            raw_detections = seg_prediction_to_items.pred_to_items(
                comb_pred=pred['mask'][0, 0].cpu().numpy(),
                offset=pred['offset'][0].cpu().numpy(),
                size=pred['size'][0].cpu().numpy(),
                tracking=pred['tracking'][0].cpu().numpy(),
                distance=pred['distance'][0, 0].cpu().numpy(),
                above_horizon=pred['above_horizon'][0, 0].cpu().numpy(),
                conf_threshold=0.15,
                pred_scale=8.0
            )

            # Rescale predictions from model space (2432x2048) to original image space
            scale_x = orig_w / 2432.0
            scale_y = orig_h / 2048.0
            for det in raw_detections:
                det['cx'] *= scale_x
                det['cy'] *= scale_y
                det['w'] *= scale_x
                det['h'] *= scale_y
                det['tracking'] = (det['tracking'][0] * scale_x, det['tracking'][1] * scale_y)

            # --- Tracker (now in original image space) ---
            confirmed_detections = tracker.update(raw_detections)

            # --- GT is already in original image space, no rescaling needed ---
            gt_items = frame_data.items
            
            matched_gt_indices = set()
            frame_fp = 0
            frame_tp = 0

            # Check every tracked detection against GT
            for det in confirmed_detections:
                best_dist = 40.0 # Match threshold (e.g., 50 pixels)
                best_gt_idx = -1
                
                for i, gt in enumerate(gt_items):
                    if i in matched_gt_indices:
                        continue
                        
                    dist = np.sqrt((det['cx'] - gt.cx)**2 + (det['cy'] - gt.cy)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_gt_idx = i
                
                if best_gt_idx != -1:
                    matched_gt_indices.add(best_gt_idx)
                    frame_tp += 1
                else:
                    frame_fp += 1 # It didn't match any real plane!
            
            frame_fn = len(gt_items) - len(matched_gt_indices) # Real planes we missed
            
            total_tp += frame_tp
            total_fp += frame_fp
            total_fn += frame_fn
            total_frames += 1
            
            # --- ADD VISUALIZATION HERE ---
            display_frame = cv2.cvtColor(current_frame_gray, cv2.COLOR_GRAY2BGR)

            for i, gt in enumerate(gt_items):
                if i in matched_gt_indices:
                    hit_widths.append(gt.w)
                else:
                    miss_widths.append(gt.w)

            for det in confirmed_detections:
                best_gt_idx = -1
                best_dist = 40.0
                for i, gt in enumerate(gt_items):
                    dist = np.sqrt((det['cx'] - gt.cx)**2 + (det['cy'] - gt.cy)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_gt_idx = i
                if best_gt_idx != -1:
                    detection_confs_and_widths.append((det.get('conf', 0.0), gt_items[best_gt_idx].w))

            for gt in gt_items:
                x1 = int(gt.cx - gt.w / 2)
                y1 = int(gt.cy - gt.h / 2)
                x2 = int(gt.cx + gt.w / 2)
                y2 = int(gt.cy + gt.h / 2)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_frame, f"GT d={gt.distance:.0f}m", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            for det in confirmed_detections:
                x1 = int(det['cx'] - det['w'] / 2)
                y1 = int(det['cy'] - det['h'] / 2)
                x2 = int(det['cx'] + det['w'] / 2)
                y2 = int(det['cy'] + det['h'] / 2)
                conf = int(det.get('conf', 0.0) * 100)
                track_id = det.get('track_id', '?')
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID:{track_id} {conf}%", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            out_video.write(display_frame)
            # --- END VISUALIZATION ---

            prev_frame_aligned = current_frame_gray  # carry raw frame forward; align_frames_dl aligns it next iteration
    
    out_video.release()
    print("Video saved to eval_output.mp4")


    # --- FINAL SCORE OUTPUT ---
    detection_rate = total_tp / max(1, (total_tp + total_fn))
    fppi = total_fp / max(1, total_frames)
    
    print("\n--- EVALUATION RESULTS ---")
    print(f"Total Frames Evaluated: {total_frames}")
    print(f"True Positives (Hits): {total_tp}")
    print(f"False Positives (Alarms): {total_fp}")
    print(f"False Negatives (Misses): {total_fn}")
    print(f"--------------------------")
    print(f"Empirical Detection Rate (EDR): {detection_rate:.4f} ({detection_rate*100:.1f}%)")
    print(f"False Positives Per Image (FPPI): {fppi:.5f}")
    print("--------------------------")

    save_detection_plots(hit_widths, miss_widths, detection_confs_and_widths, prefix="rgb")


def simple_ir_detector(frame_gray, prev_frame, min_area=1, max_area=200, debug_path=None):
    h, w = frame_gray.shape

    # 1. Create sky mask — sky is dark in IR, buildings are bright
    blur_big = cv2.GaussianBlur(frame_gray, (101, 101), 0)
    sky_mask = (blur_big < 40).astype(np.uint8) * 255

    # Exclude the black vignette corners (pixels near 0)
    _, vignette_mask = cv2.threshold(frame_gray, 3, 255, cv2.THRESH_BINARY)
    sky_mask = cv2.bitwise_and(sky_mask, vignette_mask)

    # Erode sky mask to avoid building edges
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sky_mask = cv2.erode(sky_mask, kernel_big)

    # 2. Within sky, find bright anomalies using local contrast
    blur_local = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    local_contrast = cv2.subtract(frame_gray, blur_local)  # bright spots relative to local bg

    # Only keep spots in sky region
    local_contrast = cv2.bitwise_and(local_contrast, local_contrast, mask=sky_mask)

    # Threshold — drone is a few intensity levels above local sky
    _, spot_mask = cv2.threshold(local_contrast, 3, 255, cv2.THRESH_BINARY)

    # 3. Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, kernel)

    # Save debug image
    if debug_path:
        debug = np.hstack([
            frame_gray,
            sky_mask,
            cv2.normalize(local_contrast, None, 0, 255, cv2.NORM_MINMAX),
            spot_mask
        ])
        cv2.imwrite(debug_path, debug)

    # 4. Find contours
    contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w_box, h_box = cv2.boundingRect(c)
            # Peak brightness of this spot
            roi = frame_gray[y:y+h_box, x:x+w_box]
            peak = float(roi.max())
            detections.append({
                'cx': x + w_box / 2,
                'cy': y + h_box / 2,
                'w': float(w_box),
                'h': float(h_box),
                'conf': peak / 255.0,
                'tracking': (0.0, 0.0)
            })

    if len(detections) > 0 and prev_frame is not None:
        diff = cv2.absdiff(frame_gray, prev_frame)
        global_motion = diff.mean()
        moving_detections = []
        for det in detections:
            cx, cy = int(det['cx']), int(det['cy'])
            r = 5
            y1 = max(0, cy - r)
            y2 = min(h, cy + r)
            x1 = max(0, cx - r)
            x2 = min(w, cx + r)
            local_motion = diff[y1:y2, x1:x2].mean()
            # Keep if motion is above the global average (moving more than background)
            if local_motion > max(0.2, global_motion * 1.5):
                det['motion'] = float(local_motion)
                moving_detections.append(det)
        detections = moving_detections

    return detections


def evaluate_ir_classical(ir_video_path, ir_fps=9.0, rgb_fps=30.0):
    # --- Load RGB GT dataset ---
    # We need a minimal cfg just to load the dataset
    experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")
    cfg = common_utils.load_config_data("120_hrnet32_all", experiments_dir=experiments_dir)

    val_dataset = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_VALID,
        cfg_data=cfg,
        return_torch_tensors=False,
        small_subset=False
    )

    # Index GT by frame number
    rgb_gt_by_frame_num = {}
    for i in range(len(val_dataset)):
        frame = val_dataset.frames[i]
        rgb_gt_by_frame_num[frame.frame_num] = frame.items

    # Coordinate scaling RGB -> IR
    rgb_w, rgb_h = 2496, 2048
    ir_w, ir_h = 640, 512
    scale_x = ir_w / rgb_w
    scale_y = ir_h / rgb_h

    # Empirical vertical offset between RGB GT (scaled) and IR detections
    # The IR camera is mounted lower, so GT y-coordinates need to be shifted down
    ir_y_offset = 95.0  # pixels in IR space — tune based on output
    ir_x_offset = 0.0

    
    # Open IR video
    cap = cv2.VideoCapture(ir_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter("ir_classical_output.mp4", fourcc, ir_fps, (ir_w, ir_h), isColor=True)

    tracker = DelayTracker(min_steps=5, max_distance=30.0, max_coast_frames=2)

    prev_frame = None
    ir_frame_idx = 0

    hit_widths = []
    miss_widths = []
    detection_confs_and_widths = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_frames = 0

    print("Running IR classical detection...")

    while True:
        ret, frame_color = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

        if prev_frame is None:
            prev_frame = frame_gray
            ir_frame_idx += 1
            continue

        # --- Simple IR detection (no neural network) ---
        # Replace the raw_detections line with:
        debug_path = f"ir_debug_{ir_frame_idx:04d}.png" if ir_frame_idx in [100, 500, 1000] else None
        raw_detections = simple_ir_detector(frame_gray, prev_frame, debug_path=debug_path)


        # --- Tracker ---
        confirmed_detections = tracker.update(raw_detections)

        # --- Map to RGB GT ---
        ir_time = ir_frame_idx / ir_fps
        rgb_frame_num = round(ir_time * rgb_fps)

        gt_items_rgb = []
        closest_frame = None
        if rgb_gt_by_frame_num:
            closest_frame = min(rgb_gt_by_frame_num.keys(), key=lambda f: abs(f - rgb_frame_num))
            if abs(closest_frame - rgb_frame_num) <= 2:
                gt_items_rgb = rgb_gt_by_frame_num[closest_frame]

        # Scale GT to IR space
        gt_items_ir = []
        for gt in gt_items_rgb:
            gt_items_ir.append(dataset_tracking.DetectionItem(
                cls_name=gt.cls_name,
                item_id=gt.item_id,
                distance=gt.distance,
                cx=gt.cx * scale_x + ir_x_offset,
                cy=gt.cy * scale_y + ir_y_offset,
                w=gt.w * scale_x,
                h=gt.h * scale_y,
                above_horizon=gt.above_horizon
            ))

        # --- Metrics ---
        matched_gt_indices = set()
        frame_tp = 0
        frame_fp = 0

        for det in confirmed_detections:
            best_dist = 120.0
            best_gt_idx = -1
            for i, gt in enumerate(gt_items_ir):
                if i in matched_gt_indices:
                    continue
                dist = np.sqrt((det['cx'] - gt.cx)**2 + (det['cy'] - gt.cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = i
            if best_gt_idx != -1:
                matched_gt_indices.add(best_gt_idx)
                frame_tp += 1
            else:
                frame_fp += 1

        frame_fn = len(gt_items_ir) - len(matched_gt_indices)
        total_tp += frame_tp
        total_fp += frame_fp
        total_fn += frame_fn
        total_frames += 1

        for i, gt in enumerate(gt_items_ir):
            if i in matched_gt_indices:
                hit_widths.append(gt.w)
            else:
                miss_widths.append(gt.w)

        for det in confirmed_detections:
            best_gt_idx = -1
            best_dist = 30.0
            for i, gt in enumerate(gt_items_ir):
                dist = np.sqrt((det['cx'] - gt.cx)**2 + (det['cy'] - gt.cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = i
            if best_gt_idx != -1:
                detection_confs_and_widths.append((det.get('conf', 0.0), gt_items_ir[best_gt_idx].w))

        # --- Diagnostic ---
        if ir_frame_idx % 100 == 0:
            gt_str = ""
            if gt_items_ir and confirmed_detections:
                gt = gt_items_ir[0]
                det = confirmed_detections[0]
                dist = np.sqrt((det['cx'] - gt.cx)**2 + (det['cy'] - gt.cy)**2)
                gt_str = f" | GT:({gt.cx:.0f},{gt.cy:.0f}) Det:({det['cx']:.0f},{det['cy']:.0f}) Dist:{dist:.0f}px"
            print(f"Frame {ir_frame_idx}: {len(raw_detections)} raw, {len(confirmed_detections)} confirmed, "
                  f"{len(gt_items_ir)} GT{gt_str}")

        # --- Visualization ---
        display_frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

        for gt in gt_items_ir:
            x1, y1 = int(gt.cx - gt.w / 2), int(gt.cy - gt.h / 2)
            x2, y2 = int(gt.cx + gt.w / 2), int(gt.cy + gt.h / 2)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(display_frame, "GT", (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        for det in confirmed_detections:
            x1, y1 = int(det['cx'] - det['w'] / 2), int(det['cy'] - det['h'] / 2)
            x2, y2 = int(det['cx'] + det['w'] / 2), int(det['cy'] + det['h'] / 2)
            conf = int(det.get('conf', 0.0) * 100)
            track_id = det.get('track_id', '?')
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(display_frame, f"ID:{track_id} {conf}%", (x1, y2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        out_video.write(display_frame)
        prev_frame = frame_gray
        ir_frame_idx += 1

    cap.release()
    out_video.release()

    detection_rate = total_tp / max(1, (total_tp + total_fn))
    fppi = total_fp / max(1, total_frames)

    print(f"\n--- IR CLASSICAL EVALUATION RESULTS ---")
    print(f"Total Frames Evaluated: {total_frames}")
    print(f"True Positives (Hits): {total_tp}")
    print(f"False Positives (Alarms): {total_fp}")
    print(f"False Negatives (Misses): {total_fn}")
    print(f"Empirical Detection Rate (EDR): {detection_rate:.4f} ({detection_rate*100:.1f}%)")
    print(f"False Positives Per Image (FPPI): {fppi:.5f}")
    print("Video saved to ir_classical_output.mp4")

    save_detection_plots(hit_widths, miss_widths, detection_confs_and_widths, prefix="ir_classical")


def evaluate_on_ir(experiment_name, tracking_weights, transform_weights, ir_video_path, rgb_fps=30.0, ir_fps=9.0):
    # --- Load models (same as evaluate_tracker) ---
    cfg = common_utils.load_config_data(experiment_name)
    tracking_model = build_model(cfg)
    tracking_model = tracking_model.cuda()
    tracking_model.eval()
    checkpoint = torch.load(tracking_weights, weights_only=False)
    tracking_model.load_state_dict(checkpoint["model_state_dict"])

    experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")
    tr_cfg = common_utils.load_config_data("030_tr_tsn_rn34_w3_crop_borders", experiments_dir=experiments_dir)
    transform_model = build_transform_model(tr_cfg).cuda()
    tr_checkpoint = torch.load(transform_weights, weights_only=False)
    transform_model.load_state_dict(tr_checkpoint["model_state_dict"])
    transform_model.eval()

    # --- Load RGB GT dataset ---
    val_dataset = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_VALID,
        cfg_data=cfg,
        return_torch_tensors=False,
        small_subset=False
    )

    # Build a lookup: rgb_frame_index -> gt_items
    # Assumes single flight for now
    rgb_gt_by_frame_num = {}
    for i in range(len(val_dataset)):
        frame = val_dataset.frames[i]
        rgb_gt_by_frame_num[frame.frame_num] = frame.items

    # --- Coordinate scaling from RGB to IR ---
    rgb_w, rgb_h = 2496, 2048  # adjust if different
    ir_w, ir_h = 640, 512
    scale_x = ir_w / rgb_w
    scale_y = ir_h / rgb_h

    # --- Open IR video ---
    cap = cv2.VideoCapture(ir_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter("ir_eval_output.mp4", fourcc, ir_fps, (ir_w, ir_h), isColor=True)

    tracker = DelayTracker(min_steps=8, max_distance=20.0, max_coast_frames=3)  # smaller max_distance for lower res

    prev_frame = None
    ir_frame_idx = 0

    hit_widths = []
    miss_widths = []
    detection_confs_and_widths = [] 

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_frames = 0

    print("Running IR evaluation...")

    while True:
        ret, frame_color = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        display_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

        frame_gray = 255 - frame_gray

        # Upscale IR to RGB resolution so the models work correctly
        frame_gray = cv2.resize(frame_gray, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)
        if prev_frame is None:
            prev_frame = frame_gray
            ir_frame_idx += 1
            continue

        # --- Map IR frame to closest RGB frame ---
        ir_time = ir_frame_idx / ir_fps
        rgb_frame_num = round(ir_time * rgb_fps)

        # Find closest available GT frame
        if rgb_gt_by_frame_num:
            closest_frame = min(rgb_gt_by_frame_num.keys(), key=lambda f: abs(f - rgb_frame_num))
            # Only use GT if within 2 RGB frames of alignment
            if abs(closest_frame - rgb_frame_num) <= 2:
                gt_items_rgb = rgb_gt_by_frame_num[closest_frame]
            else:
                gt_items_rgb = []
        else:
            gt_items_rgb = []

        # --- Run model ---
        prev_aligned = prev_frame
        tensor_inputs = prepare_inputs(frame_gray, prev_aligned, target_size=(ir_h, ir_w))
        with torch.cuda.amp.autocast(), torch.no_grad():
            pred = tracking_model(tensor_inputs)

        pred['mask'] = torch.sigmoid(pred['mask'])

        raw_detections = seg_prediction_to_items.pred_to_items(
            comb_pred=pred['mask'][0, 0].cpu().numpy(),
            offset=pred['offset'][0].cpu().numpy(),
            size=pred['size'][0].cpu().numpy(),
            tracking=pred['tracking'][0].cpu().numpy(),
            distance=pred['distance'][0, 0].cpu().numpy(),
            above_horizon=pred['above_horizon'][0, 0].cpu().numpy(),
            conf_threshold=0.01,
            pred_scale=8.0
        )
        
        if ir_frame_idx % 100 == 0:
            mask_val = pred['mask'][0, 0].cpu().numpy()
            print(f"Frame {ir_frame_idx}: mask min={mask_val.min():.4f} max={mask_val.max():.4f} "
                    f"mean={mask_val.mean():.6f} | {len(raw_detections)} raw, {len(confirmed_detections)} confirmed")


        # Rescale predictions from model space (2432x2048) to RGB space
        model_scale_x = rgb_w / 2432.0
        model_scale_y = rgb_h / 2048.0
        for det in raw_detections:
            det['cx'] = det['cx'] * model_scale_x * scale_x  # RGB -> IR
            det['cy'] = det['cy'] * model_scale_y * scale_y
            det['w'] *= model_scale_x * scale_x
            det['h'] *= model_scale_y * scale_y
            det['tracking'] = (det['tracking'][0] * model_scale_x * scale_x,
                               det['tracking'][1] * model_scale_y * scale_y)

        confirmed_detections = tracker.update(raw_detections)


        # --- Scale RGB GT to IR space ---
        gt_items_ir = []
        for gt in gt_items_rgb:
            gt_items_ir.append(dataset_tracking.DetectionItem(
                cls_name=gt.cls_name,
                item_id=gt.item_id,
                distance=gt.distance,
                cx=gt.cx * scale_x,
                cy=gt.cy * scale_y,
                w=gt.w * scale_x,
                h=gt.h * scale_y,
                above_horizon=gt.above_horizon
            ))
        
        if ir_frame_idx % 100 == 0:
            print(f"Frame {ir_frame_idx}: IR time={ir_time:.2f}s -> RGB frame {rgb_frame_num}, "
                f"closest GT frame={closest_frame}, {len(gt_items_ir)} GT items, "
                f"{len(confirmed_detections)} confirmed dets")

        # --- Metrics ---
        matched_gt_indices = set()
        frame_tp = 0
        frame_fp = 0

        for det in confirmed_detections:
                best_gt_idx = -1
                best_dist = 20.0
                for i, gt in enumerate(gt_items_ir):
                    dist = np.sqrt((det['cx'] - gt.cx)**2 + (det['cy'] - gt.cy)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_gt_idx = i
                if best_gt_idx != -1:
                    detection_confs_and_widths.append((det.get('conf', 0.0), gt_items_ir[best_gt_idx].w))

        frame_fn = len(gt_items_ir) - len(matched_gt_indices)
        total_tp += frame_tp
        total_fp += frame_fp
        total_fn += frame_fn
        total_frames += 1

        for i, gt in enumerate(gt_items_ir):
            if i in matched_gt_indices:
                hit_widths.append(gt.w)
            else:
                miss_widths.append(gt.w)

        if ir_frame_idx % 100 == 0 and gt_items_ir and confirmed_detections:
                gt = gt_items_ir[0]
                det = confirmed_detections[0]
                dist = np.sqrt((det['cx'] - gt.cx)**2 + (det['cy'] - gt.cy)**2)
                print(f"  GT: ({gt.cx:.1f}, {gt.cy:.1f})  Det: ({det['cx']:.1f}, {det['cy']:.1f})  Dist: {dist:.1f}px")

        # --- Visualization ---
        display_frame = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)

        for gt in gt_items_ir:
            x1, y1 = int(gt.cx - gt.w / 2), int(gt.cy - gt.h / 2)
            x2, y2 = int(gt.cx + gt.w / 2), int(gt.cy + gt.h / 2)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(display_frame, f"GT", (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        for det in confirmed_detections:
            x1, y1 = int(det['cx'] - det['w'] / 2), int(det['cy'] - det['h'] / 2)
            x2, y2 = int(det['cx'] + det['w'] / 2), int(det['cy'] + det['h'] / 2)
            conf = int(det.get('conf', 0.0) * 100)
            track_id = det.get('track_id', '?')
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(display_frame, f"ID:{track_id} {conf}%", (x1, y2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        out_video.write(display_frame)
        prev_frame = frame_gray
        ir_frame_idx += 1

    cap.release()
    out_video.release()

    detection_rate = total_tp / max(1, (total_tp + total_fn))
    fppi = total_fp / max(1, total_frames)

    print(f"\n--- IR EVALUATION RESULTS ---")
    print(f"Total Frames Evaluated: {total_frames}")
    print(f"True Positives (Hits): {total_tp}")
    print(f"False Positives (Alarms): {total_fp}")
    print(f"False Negatives (Misses): {total_fn}")
    print(f"Empirical Detection Rate (EDR): {detection_rate:.4f} ({detection_rate*100:.1f}%)")
    print(f"False Positives Per Image (FPPI): {fppi:.5f}")
    print("Video saved to ir_eval_output.mp4")

    save_detection_plots(hit_widths, miss_widths, detection_confs_and_widths, prefix="ir")

def save_detection_plots(hit_widths, miss_widths, detection_confs_and_widths, prefix="rgb"):
    import matplotlib.pyplot as plt

    bins = np.arange(0, max(hit_widths + miss_widths + [1]) + 5, 5)

    # 1. Hit vs Miss histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(miss_widths, bins=bins, alpha=0.6, color='red', label=f'Missed ({len(miss_widths)})')
    ax.hist(hit_widths, bins=bins, alpha=0.6, color='green', label=f'Detected ({len(hit_widths)})')
    ax.set_xlabel('GT Bounding Box Width (px)')
    ax.set_ylabel('Count')
    ax.set_title(f'[{prefix.upper()}] Detection Performance by Object Size')
    ax.legend()
    plt.savefig(f'{prefix}_detection_by_size.png', dpi=150)
    plt.close()
    print(f"Plot saved to {prefix}_detection_by_size.png")

    # 2. Detection rate by size
    hit_arr = np.array(hit_widths)
    miss_arr = np.array(miss_widths)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hit_counts, _ = np.histogram(hit_arr, bins=bins)
    total_counts, _ = np.histogram(np.concatenate([hit_arr, miss_arr]), bins=bins)
    rate = np.where(total_counts > 0, hit_counts / total_counts, 0)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(bin_centers, rate, width=4, color='steelblue')
    ax2.set_xlabel('GT Bounding Box Width (px)')
    ax2.set_ylabel('Detection Rate')
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f'[{prefix.upper()}] Detection Rate by Object Size')
    plt.savefig(f'{prefix}_detection_rate_by_size.png', dpi=150)
    plt.close()
    print(f"Plot saved to {prefix}_detection_rate_by_size.png")

    # 3. Confidence vs size with trendline
    if detection_confs_and_widths:
        confs, widths = zip(*detection_confs_and_widths)
        confs = np.array(confs, dtype=np.float64)
        widths = np.array(widths, dtype=np.float64)

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.scatter(widths, confs, alpha=0.3, s=10, color='steelblue', label='Detections')

        # Trendline (polynomial degree 2)
        if len(widths) > 3:
            sort_idx = np.argsort(widths)
            widths_sorted = widths[sort_idx]
            confs_sorted = confs[sort_idx]
            coeffs = np.polyfit(widths_sorted, confs_sorted, 2)
            trend_x = np.linspace(widths_sorted.min(), widths_sorted.max(), 200)
            trend_y = np.clip(np.polyval(coeffs, trend_x), 0, 1)
            ax3.plot(trend_x, trend_y, color='red', linewidth=2, label='Trendline')

        ax3.set_xlabel('GT Bounding Box Width (px)')
        ax3.set_ylabel('Confidence Score')
        ax3.set_ylim(0, 1.05)
        ax3.set_title(f'[{prefix.upper()}] Detection Confidence vs Object Size')
        ax3.legend()
        plt.savefig(f'{prefix}_confidence_vs_size.png', dpi=150)
        plt.close()
        print(f"Plot saved to {prefix}_confidence_vs_size.png")


def align_frames_dl(transform_model, cur_frame_full, prev_frame_full):
    """
    Uses the trained TSM model to calculate the transformation and align the previous frame.
    """
    full_h, full_w = cur_frame_full.shape
    
    # 1. Take the 1024x1024 center crop as the model expects
    crop_h, crop_w = 1024, 1024
    y0 = (full_h - crop_h) // 2
    x0 = (full_w - crop_w) // 2
    
    cur_crop = cur_frame_full[y0:y0+crop_h, x0:x0+crop_w].astype(np.float32) / 255.0
    prev_crop = prev_frame_full[y0:y0+crop_h, x0:x0+crop_w].astype(np.float32) / 255.0

    cur_tensor = torch.from_numpy(cur_crop).unsqueeze(0).cuda()
    prev_tensor = torch.from_numpy(prev_crop).unsqueeze(0).cuda()

    # 2. Predict the offsets and heatmap weights
    with torch.amp.autocast(device_type='cuda'), torch.no_grad():
        heatmap, offsets = transform_model(prev_tensor, cur_tensor)

    heatmap = heatmap[0].cpu().numpy()
    offsets = offsets[0].cpu().numpy()

    # 3. Generate the author's 32-pixel spaced tracking grid
    prev_points = np.zeros((2, 32, 32), dtype=np.float32)
    prev_points[0, :, :] = np.arange(16, 1024, 32)[None, :]
    prev_points[1, :, :] = np.arange(16, 1024, 32)[:, None]
    prev_points = prev_points[..., 2:-2, 2:-2] # Slices down to (2, 28, 28)

    cur_points = prev_points + offsets
    center_offset = np.array([512.0, 512.0])[:, None]

    # 4. Extract global dx, dy, and angle from the grid
    dx, dy, angle, err = offset_grid_to_transform.offset_grid_to_transform_params(
        prev_frame_points=prev_points.reshape(2, -1) - center_offset,
        cur_frame_points=cur_points.reshape(2, -1) - center_offset,
        points_weight=heatmap.reshape(-1) ** 2
    )

    # 5. Build the transformation matrix for the FULL 4K frame
    full_tr = common_utils.build_geom_transform(
        dst_w=full_w,
        dst_h=full_h,
        src_center_x=full_w / 2 + dx,
        src_center_y=full_h / 2 + dy,
        scale_x=1.0,
        scale_y=1.0,
        angle=angle,
        return_params=True
    )

    # 6. Warp the full previous frame to match the current frame
    aligned_prev_frame = cv2.warpAffine(
        prev_frame_full,
        full_tr[:2, :], 
        dsize=(full_w, full_h),
        flags=cv2.INTER_LINEAR
    )

    return aligned_prev_frame

def prepare_inputs(current_frame, prev_frame_aligned, target_size=(2048, 2432)):
    # Resizing ensures compatibility with the HRNet32 backbone trained on challenge data [cite: 10]
    if current_frame.shape[:2] != target_size:
        current_frame = cv2.resize(current_frame, (target_size[1], target_size[0]))
        prev_frame_aligned = cv2.resize(prev_frame_aligned, (target_size[1], target_size[0]))
    
    # ... rest of the normalization logic ...
    # 2. Normalize and convert to float32 [cite: 10]
    curr_norm = current_frame.astype(np.float32) / 255.0
    prev_norm = prev_frame_aligned.astype(np.float32) / 255.0

    # 3. Stack and add batch dimension [cite: 77]
    curr_tensor = torch.from_numpy(curr_norm)
    prev_tensor = torch.from_numpy(prev_norm)
    stacked = torch.stack([prev_tensor, curr_tensor], dim=0).unsqueeze(0)

    return stacked.cuda()

def run_inference(experiment_name, model_weights_path, video_path, output_path="output_tracked.mp4"):
    # 2. LOAD YOUR TRAINED TRACKING MODEL
    cfg = common_utils.load_config_data(experiment_name)
    tracking_model = build_model(cfg)
    tracking_model = tracking_model.cuda()
    tracking_model.eval()
    
    # Load the weights exported by train.py
    checkpoint = torch.load(model_weights_path)
    tracking_model.load_state_dict(checkpoint["model_state_dict"])

    # LOAD THE TSM TRANSFORMATION MODEL
    # Ensure you use the exact experiment name that matches your config file for '030_tr_tsn_rn34'
    experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")
    tr_cfg = common_utils.load_config_data("030_tr_tsn_rn34_w3_crop_borders", experiments_dir=experiments_dir)
    transform_model = build_transform_model(tr_cfg).cuda()
    
    # Load your trained weights for the transform model
    tr_checkpoint = torch.load("/cluster/home/nbaruffol/airborne_detection/output/checkpoints/030_tr_tsn_rn34_w3_crop_borders/0/500.pt") # UPDATE THIS PATH
    transform_model.load_state_dict(tr_checkpoint["model_state_dict"])
    transform_model.eval()
    
    # 3. INITIALIZE YOUR TRACKER
    # Distance threshold of ~40 pixels is what the author recommended [cite: 1]
    tracker = DelayTracker(min_steps=8, max_distance=40.0, max_coast_frames=5)
    
    # 1. SETUP VIDEO READER AND WRITER
    cap = cv2.VideoCapture(video_path)
    
    # Get original video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Setup the writer (MP4 format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Pad frames to the model's expected input size (trained on 2448x2048)
    model_w, model_h = 2560, 2048
    x_pad = (model_w - width) // 2
    y_pad = (model_h - height) // 2

    # Read first frame to initialize
    ret, prev_frame_color = cap.read()
    if not ret:
        print("Failed to read video")
        return

    prev_frame_raw = cv2.cvtColor(prev_frame_color, cv2.COLOR_BGR2GRAY)
    prev_frame = np.zeros((model_h, model_w), dtype=np.uint8)
    prev_frame[y_pad:y_pad+height, x_pad:x_pad+width] = prev_frame_raw

    print("Starting inference...")
    
    # 2. THE MAIN LOOP
    while True:
        ret, current_frame_color = cap.read()
        if not ret: 
            break # End of video
        
        current_frame_raw = cv2.cvtColor(current_frame_color, cv2.COLOR_BGR2GRAY)
        current_frame_gray = np.zeros((model_h, model_w), dtype=np.uint8)
        current_frame_gray[y_pad:y_pad+height, x_pad:x_pad+width] = current_frame_raw

        # --- Network Prediction (from previous step) ---
        prev_frame_aligned = align_frames_dl(transform_model, current_frame_gray, prev_frame)
        tensor_inputs = prepare_inputs(current_frame_gray, prev_frame_aligned)
        
        with torch.amp.autocast(device_type='cuda'), torch.no_grad():
            pred = tracking_model(tensor_inputs)
            
        pred['mask'] = torch.sigmoid(pred['mask'])
        
        raw_detections = seg_prediction_to_items.pred_to_items(
            comb_pred=pred['mask'][0, 0].cpu().numpy(),
            offset=pred['offset'][0].cpu().numpy(),
            size=pred['size'][0].cpu().numpy(),
            tracking=pred['tracking'][0].cpu().numpy(),
            distance=pred['distance'][0, 0].cpu().numpy(),
            above_horizon=pred['above_horizon'][0, 0].cpu().numpy(),
            conf_threshold=0.15,
            pred_scale=8.0,
            x_offset=-x_pad,
            y_offset=-y_pad,
        )
        orig_h, orig_w = current_frame_gray.shape
        scale_x = orig_w / 2432.0
        scale_y = orig_h / 2048.0
        for det in raw_detections:
            det['cx'] *= scale_x
            det['cy'] *= scale_y
            det['w'] *= scale_x
            det['h'] *= scale_y
            det['tracking'] = (det['tracking'][0] * scale_x, det['tracking'][1] * scale_y)
        
        # --- Run the Tracker ---
        confirmed_detections = tracker.update(raw_detections)

        # 3. --- VISUALIZATION ---
        # We draw on 'current_frame_color' so we can have colored boxes
        display_frame = current_frame_color.copy()

        for det in confirmed_detections:
            # Extract the tracked data
            cx, cy = det['cx'], det['cy']
            w, h = det['w'], det['h']
            track_id = det.get('track_id', 'Unknown')
            conf = int(det.get('conf', 0.0) * 100) # Convert to percentage

            # Skip detections that fell in the black padding border
            if cx < 0 or cx > width or cy < 0 or cy > height:
                continue

            # Convert Center (cx, cy) to Top-Left and Bottom-Right
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            # Color: BGR format (Green is 0, 255, 0)
            box_color = (0, 255, 0) 
            text_color = (0, 0, 0) # Black text
            bg_color = (0, 255, 0) # Green background for text

            # Draw the bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Prepare the label text (e.g., "ID:4 85%")
            label = f"ID:{track_id} {conf}%"
            
            # Optional: Draw a solid background rectangle for the text so it's readable against clouds
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), bg_color, -1)
            
            # Draw the text
            cv2.putText(display_frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Write the finished frame to the output video
        out_video.write(display_frame)
        
        # Update previous frame for the next loop iteration
        prev_frame = current_frame_gray

    # Cleanup
    cap.release()
    out_video.release()
    print(f"Finished! Video saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="visualize",
                        choices=["visualize", "evaluate_rgb", "evaluate_ir", "evaluate_ir_classical"])
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--ir_video", type=str, help="Path to IR video for evaluate_ir mode")
    parser.add_argument("--exp", type=str, default="my_hrnet_experiment", help="Experiment name")
    parser.add_argument("--model_weights", type=str,
                        default="/cluster/home/nbaruffol/airborne_detection/output/checkpoints/120_hrnet32_all/0/2220.pt")
    parser.add_argument("--tracking_weights", type=str,
                        default="/cluster/home/nbaruffol/airborne_detection/output/checkpoints/120_hrnet32_all/0/2220.pt")
    parser.add_argument("--transform_weights", type=str,
                        default="/cluster/home/nbaruffol/airborne_detection/output/checkpoints/030_tr_tsn_rn34_w3_crop_borders/0/500.pt")
    args = parser.parse_args()

    if args.mode == "visualize":
        run_inference(
            experiment_name=args.exp,
            model_weights_path=args.model_weights,
            video_path=args.video
        )
    elif args.mode == "evaluate_rgb":
        evaluate_tracker(
            experiment_name=args.exp,
            tracking_weights=args.tracking_weights,
            model_weights_path=args.model_weights,
            transform_weights=args.transform_weights,
        )
    elif args.mode == "evaluate_ir":
        evaluate_on_ir(
            experiment_name=args.exp,
            tracking_weights=args.tracking_weights,
            transform_weights=args.transform_weights,
            ir_video_path=args.ir_video,
        )
    elif args.mode == "evaluate_ir_classical":
        evaluate_ir_classical(
            ir_video_path=args.ir_video,
        )