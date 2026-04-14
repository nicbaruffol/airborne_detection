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

def evaluate_tracker(output_dir, experiment_name, tracking_weights, model_weights_path, transform_weights):
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

    if len(val_dataset.frames) == 0:
        print("No validation frames found. Dataset is empty (missing groundtruth.csv).")
        print("\n--- EVALUATE RGB RESULTS ---")
        print("Total Frames Evaluated: 0")
        print("True Positives (Hits): 0")
        print("False Positives (Alarms): 0")
        print("False Negatives (Misses): 0")
        print("Empirical Detection Rate (EDR): 0.0000 (0.0%)")
        print("False Positives Per Image (FPPI): 0.00000")
        print("All done!")
        return

    sample_frame = cv2.imread(
        val_dataset.img_fn(val_dataset.frames[0].part, val_dataset.frames[0].flight_id, val_dataset.frames[0].img_name),
        cv2.IMREAD_GRAYSCALE
    )
    vid_h, vid_w = sample_frame.shape

    fps = 30.0 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(os.path.join(output_dir,"eval_output.mp4"), fourcc, fps, (vid_w, vid_h), isColor=True)

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
            with torch.amp.autocast('cuda'), torch.no_grad():
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

            # --- Metrics & Plot Data Collection ---
            for det in confirmed_detections:
                best_dist = 50.0  # Adjust this depending on the function (e.g., 40.0 for IR, 100.0 for RGB)
                best_gt_idx = -1
                
                for i, gt in enumerate(gt_items):
                    if i in matched_gt_indices:
                        continue
                    
                    # Safely handle both dicts (CSV mode) and objects (Fused/Classical mode)
                    gt_cx = gt['cx'] if isinstance(gt, dict) else gt.cx
                    gt_cy = gt['cy'] if isinstance(gt, dict) else gt.cy
                    gt_w = gt['w'] if isinstance(gt, dict) else gt.w
                    gt_h = gt['h'] if isinstance(gt, dict) else gt.h
                        
                    dist = np.sqrt((det['cx'] - gt_cx)**2 + (det['cy'] - gt_cy)**2)
                    
                    # Calculate how much bigger or smaller the detection is compared to GT
                    width_ratio = det['w'] / max(1, gt_w)
                    height_ratio = det['h'] / max(1, gt_h)
                    
                    # Only count it as a match IF it's close AND the size is within your custom 20% to 200% range
                    is_right_size = (0.2 < width_ratio < 3.0) and (0.2 < height_ratio < 3.0)

                    if dist < best_dist and is_right_size:
                        best_dist = dist
                        best_gt_idx = i
                
                # Extract confidence score safely
                conf_val = det.get('conf', 0.0) if 'conf' in det else det['conf']
                
                if best_gt_idx != -1:
                    # --- CORRECT DETECTION (True Positive) ---
                    matched_gt_indices.add(best_gt_idx)
                    frame_tp += 1
                    
                    # Save for the scatter plot (Green)
                    matched_gt = gt_items[best_gt_idx]
                    matched_gt_w = matched_gt['w'] if isinstance(matched_gt, dict) else matched_gt.w
                    detection_confs_and_widths.append((conf_val, matched_gt_w, True))
                else:
                    # --- WRONG DETECTION (False Positive) ---
                    frame_fp += 1 
                    
                    # Save for the scatter plot (Red) - using the detection's own width since no GT matched
                    detection_confs_and_widths.append((conf_val, det['w'], False))
            
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

    save_detection_plots(output_dir, hit_widths, miss_widths, detection_confs_and_widths, prefix="rgb")


def simple_ir_detector(output_dir, frame_gray, prev_frame, min_area=1, max_area=200, debug_path=None):
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


def evaluate_yolo(yolo_weights, mode, rgb_video_path=None, ir_video_path=None,
                  output_dir="eval_results", conf_threshold=0.25,
                  rgb_fps=30.0, ir_fps=9.0):

    from ultralytics import YOLO
    from ultralytics.engine.predictor import BasePredictor
    import torch
    import os
    import cv2
    import numpy as np
    from tqdm import tqdm
    import config

    # Patch missing SilenceChannel for older exported weights
    try:
        from ultralytics.nn.modules.conv import SilenceChannel
    except (ImportError, AttributeError):
        import torch.nn as nn
        from ultralytics.nn.modules import conv
        class SilenceChannel(nn.Module):
            def __init__(self, c1=None, c2=None):
                super().__init__()
            def forward(self, x):
                return x[:, :x.shape[1]//2, :, :]
        conv.SilenceChannel = SilenceChannel
        print("  Patched missing SilenceChannel module")

    # Patch to handle 6-channel input for RGBT mode dynamically
    if mode == 'rgbt':
        original_setup_model = BasePredictor.setup_model
        def patched_setup_model(self, model, verbose=False):
            original_setup_model(self, model, verbose)
            
            # Dynamically check if the model actually has 6 channel weights
            try:
                # Get the first convolutional layer's weights
                first_conv_weight = next(self.model.parameters())
                actual_channels = first_conv_weight.shape[1]
                self.channels = actual_channels
                print(f"  [Patch] Detected model expects {actual_channels} channels.")
            except Exception as e:
                print(f"  [Patch Warning]: Could not detect channels dynamically. Forcing 6. Error: {e}")
                self.channels = 6
                
        BasePredictor.setup_model = patched_setup_model

    print(f"Loading YOLO model ({mode})...")
    
    # Initialize SAHI for RGB mode, standard YOLO for others
    if mode == 'rgb':
        from sahi.models.ultralytics import UltralyticsDetectionModel
        print("  Initializing SAHI for high-res RGB tiling...")
        model = UltralyticsDetectionModel(
            model_path=yolo_weights,
            confidence_threshold=conf_threshold,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
    else:
        model = YOLO(yolo_weights, task="detect")
        
    print(f"  Loaded: {yolo_weights}")

    # --- Load GT directly from CSV ---
    import pandas as pd
    gt_csv = f'{config.DATA_DIR}/part1/ImageSets/groundtruth.csv'
    df_gt = pd.read_csv(gt_csv)

    # Build GT lookup by frame number
    rgb_gt_by_frame_num = {}
    for _, row in df_gt.iterrows():
        frame_num = row['frame']
        if frame_num not in rgb_gt_by_frame_num:
            rgb_gt_by_frame_num[frame_num] = []

        item_id = row['id']
        if isinstance(item_id, str) or not np.isnan(item_id):
            rgb_gt_by_frame_num[frame_num].append({
                'cx': (row['gt_left'] + row['gt_right']) / 2,
                'cy': (row['gt_top'] + row['gt_bottom']) / 2,
                'w': row['gt_right'] - row['gt_left'],
                'h': row['gt_bottom'] - row['gt_top'],
                'distance': row['range_distance_m']
            })

    # --- Open videos ---
    if mode in ('rgb', 'rgbt'):
        cap_rgb = cv2.VideoCapture(rgb_video_path)
        orig_w = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cap_rgb = None

    if mode in ('ir', 'rgbt'):
        cap_ir = cv2.VideoCapture(ir_video_path)
        ir_w = int(cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH))
        ir_h = int(cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cap_ir = None

    if mode == 'rgb':
        n_frames = n_rgb
        out_fps = rgb_fps
        out_w, out_h = orig_w, orig_h
    elif mode == 'ir':
        n_frames = n_ir
        out_fps = ir_fps
        out_w, out_h = ir_w, ir_h
    else:
        n_frames = n_ir
        out_fps = ir_fps
        out_w, out_h = orig_w, orig_h

    # GT is in original image space — compute scale to video space
    # GT is already in 1440x1080 RGB video space
    if mode == 'ir':
        # Scale GT from RGB space (1440x1080) to IR space (640x512)
        gt_sx = 640.0 / 1440.0
        gt_sy = 512.0 / 1080.0
    else:
        # RGB and RGBT detections are already in 1440x1080 — no scaling
        gt_sx = 1.0
        gt_sy = 1.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(
        os.path.join(output_dir, f"yolo_{mode}_eval.mp4"),
        fourcc, out_fps, (out_w, out_h), isColor=True
    )

    hit_widths = []
    miss_widths = []
    detection_confs_and_widths = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_frames = 0

    print(f"Running YOLO {mode} evaluation ({n_frames} frames)...")

    for frame_idx in tqdm(range(n_frames)):
        # --- Read frames ---
        if mode == 'rgb':
            ret, frame_bgr = cap_rgb.read()
            if not ret: break
            display_frame = frame_bgr.copy()
            rgb_frame_num = frame_idx

        elif mode == 'ir':
            ret, frame_ir = cap_ir.read()
            if not ret: break
            display_frame = frame_ir.copy()
            rgb_frame_num = round((frame_idx / ir_fps) * rgb_fps)

        else:  # rgbt
            ret_ir, frame_ir = cap_ir.read()
            if not ret_ir: break
            rgb_idx = round((frame_idx / ir_fps) * rgb_fps)
            rgb_idx = max(0, min(rgb_idx, n_rgb - 1))
            cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, rgb_idx)
            ret_rgb, frame_bgr = cap_rgb.read()
            if not ret_rgb: break
            display_frame = frame_bgr.copy()
            rgb_frame_num = rgb_idx

        # --- Run Inference ---
        detections = []
        
        if mode == 'rgb':
            from sahi.predict import get_sliced_prediction
            # Run SAHI natively matching the gradio app logic
            result = get_sliced_prediction(
                frame_bgr,
                model, # this is the UltralyticsDetectionModel initialized earlier
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            for obj in result.object_prediction_list:
                conf = float(obj.score.value)
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = obj.bbox.to_xyxy()
                    detections.append({
                        'cx': (x1 + x2) / 2,
                        'cy': (y1 + y2) / 2,
                        'w': float(x2 - x1),
                        'h': float(y2 - y1),
                        'conf': conf,
                    })

        elif mode == 'ir':
            results = model(frame_ir, verbose=False, conf=conf_threshold)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu())
                    detections.append({
                        'cx': (x1 + x2) / 2,
                        'cy': (y1 + y2) / 2,
                        'w': float(x2 - x1),
                        'h': float(y2 - y1),
                        'conf': conf,
                    })

        else:  # rgbt
            # 1. Ensure spatial dimensions match
            if frame_ir.shape[:2] != frame_bgr.shape[:2]:
                frame_ir = cv2.resize(frame_ir, (frame_bgr.shape[1], frame_bgr.shape[0]))
            
            # 2. Combine into (H, W, 6) just like Gradio
            combined_input = np.concatenate((frame_bgr, frame_ir), axis=-1)
            
            # THE FIX: Since the model weights are [16, 3, 3, 3], we must slice it to 3 channels.
            # This replicates what the Gradio TRT engine was silently doing behind the scenes!
            if model.model.pt: # If it fell back to PyTorch
                combined_input = combined_input[..., :3]
            
            # 3. Run inference natively
            results = model(combined_input, verbose=False, conf=conf_threshold)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu())
                    detections.append({
                        'cx': (x1 + x2) / 2,
                        'cy': (y1 + y2) / 2,
                        'w': float(x2 - x1),
                        'h': float(y2 - y1),
                        'conf': conf,
                    })

        # --- GT lookup (scaled to video space) ---
        gt_items = []
        if rgb_gt_by_frame_num:
            closest_frame = min(rgb_gt_by_frame_num.keys(), key=lambda f: abs(f - rgb_frame_num))
            if abs(closest_frame - rgb_frame_num) <= 2:
                for gt in rgb_gt_by_frame_num[closest_frame]:
                    gt_items.append({
                        'cx': gt['cx'] * gt_sx,
                        'cy': gt['cy'] * gt_sy,
                        'w': gt['w'] * gt_sx,
                        'h': gt['h'] * gt_sy,
                        'distance': gt['distance']
                    })

        # In the GT scaling block, after creating gt_items:
        if mode == 'ir':
            for gt in gt_items:
                gt['cy'] += 95.0  # empirical camera offset

        # --- Metrics ---
        match_threshold = 120.0 if mode == 'ir' else 50.0
        matched_gt_indices = set()
        frame_tp = 0
        frame_fp = 0

        for det in detections:
            best_dist = match_threshold
            best_gt_idx = -1
            for i, gt in enumerate(gt_items):
                if i in matched_gt_indices:
                    continue
                dist = np.sqrt((det['cx'] - gt['cx'])**2 + (det['cy'] - gt['cy'])**2)
                # Calculate how much bigger or smaller the detection is compared to GT
                width_ratio = det['w'] / max(1, gt.w)
                height_ratio = det['h'] / max(1, gt.h)
                
                # Only count it as a match IF it's close AND the size is within 50% to 150% of GT
                is_right_size = (0.2 < width_ratio < 2) and (0.2 < height_ratio < 2)

                if dist < best_dist and is_right_size:
                    best_dist = dist
                    best_gt_idx = i
            if best_gt_idx != -1:
                matched_gt_indices.add(best_gt_idx)
                frame_tp += 1
            else:
                frame_fp += 1

        frame_fn = len(gt_items) - len(matched_gt_indices)
        total_tp += frame_tp
        total_fp += frame_fp
        total_fn += frame_fn
        total_frames += 1

        for i, gt in enumerate(gt_items):
            if i in matched_gt_indices:
                hit_widths.append(gt['w'])
            else:
                miss_widths.append(gt['w'])

        for det in detections:
            best_gt_idx = -1
            best_dist = match_threshold
            for i, gt in enumerate(gt_items):
                dist = np.sqrt((det['cx'] - gt['cx'])**2 + (det['cy'] - gt['cy'])**2)
                # Calculate how much bigger or smaller the detection is compared to GT
                width_ratio = det['w'] / max(1, gt.w)
                height_ratio = det['h'] / max(1, gt.h)
                
                # Only count it as a match IF it's close AND the size is within 50% to 150% of GT
                is_right_size = (0.2 < width_ratio < 2) and (0.2 < height_ratio < 2)

                if dist < best_dist and is_right_size:
                    best_dist = dist
                    best_gt_idx = i
            if best_gt_idx != -1:
                detection_confs_and_widths.append((det['conf'], gt_items[best_gt_idx]['w']))

        # --- Visualization ---
        for gt in gt_items:
            x1, y1 = int(gt['cx'] - gt['w'] / 2), int(gt['cy'] - gt['h'] / 2)
            x2, y2 = int(gt['cx'] + gt['w'] / 2), int(gt['cy'] + gt['h'] / 2)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display_frame, "GT", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        for det in detections:
            x1, y1 = int(det['cx'] - det['w'] / 2), int(det['cy'] - det['h'] / 2)
            x2, y2 = int(det['cx'] + det['w'] / 2), int(det['cy'] + det['h'] / 2)
            conf = int(det['conf'] * 100)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{conf}%", (x1, y2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        out_video.write(display_frame)

        if frame_idx % 100 == 0:
            gt_str = ""
            if gt_items and detections:
                gt = gt_items[0]
                det = detections[0]
                dist = np.sqrt((det['cx'] - gt['cx'])**2 + (det['cy'] - gt['cy'])**2)
                gt_str = f" | GT:({gt['cx']:.0f},{gt['cy']:.0f}) Det:({det['cx']:.0f},{det['cy']:.0f}) Dist:{dist:.0f}px"
            print(f"Frame {frame_idx}: {len(detections)} dets, {len(gt_items)} GT{gt_str}")

    if cap_rgb: cap_rgb.release()
    if cap_ir: cap_ir.release()
    out_video.release()

    detection_rate = total_tp / max(1, (total_tp + total_fn))
    fppi = total_fp / max(1, total_frames)

    print(f"\n--- YOLO {mode.upper()} EVALUATION RESULTS ---")
    print(f"Total Frames Evaluated: {total_frames}")
    print(f"True Positives (Hits): {total_tp}")
    print(f"False Positives (Alarms): {total_fp}")
    print(f"False Negatives (Misses): {total_fn}")
    print(f"Empirical Detection Rate (EDR): {detection_rate:.4f} ({detection_rate*100:.1f}%)")
    print(f"False Positives Per Image (FPPI): {fppi:.5f}")
    print(f"Video saved to {os.path.join(output_dir, f'yolo_{mode}_eval.mp4')}")

    save_detection_plots(output_dir, hit_widths, miss_widths, detection_confs_and_widths, prefix=f"yolo_{mode}")


def plot_comparison(results_dict, output_dir):
    """
    Plots the final comparative bar chart between RGB, IR, and Fused pipelines.
    Expects results_dict format: {'rgb': {'edr': 0.05, 'fppi': 0.1}, ...}
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    modes = list(results_dict.keys())
    
    # Extract metrics, converting EDR to a percentage
    edrs = [results_dict[m]['edr'] * 100 for m in modes]
    fppis = [results_dict[m]['fppi'] for m in modes]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Bar chart for EDR (Left Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Model Pipeline')
    ax1.set_ylabel('Empirical Detection Rate (%)', color=color)
    bars1 = ax1.bar(x - width/2, edrs, width, color=color, label='Detection Rate (%)')
    ax1.tick_params(axis='y', labelcolor=color)

    # Bar chart for FPPI (Right Axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('False Positives Per Image (FPPI)', color=color)
    bars2 = ax2.bar(x + width/2, fppis, width, color=color, label='FPPI')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Performance Comparison: RGB vs IR vs Fused')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in modes])
    
    # Combined legend
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    
    save_path = os.path.join(output_dir, "yolo_csv_comparison.png")
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}\n")
    plt.close()

def evaluate_yolo_csv(csv_path, mode, output_dir="eval_results", rgb_fps=30.0, ir_fps=8.57):
    import pandas as pd
    import numpy as np
    import config

    print(f"Loading YOLO {mode} detections from {csv_path}...")
    det_df = pd.read_csv(csv_path)
    
    gt_csv = f'{config.DATA_DIR}/part1/ImageSets/groundtruth.csv'
    df_gt = pd.read_csv(gt_csv)
    
    max_gt_frame = df_gt['frame'].max()
    
    # --- EXPECTED FRAMES LOGIC ---
    if mode == 'rgb':
        expected_gt_frames = list(range(max_gt_frame + 1))
    elif mode == 'rgbt':
        expected_gt_frames = list(range(1601)) 
    elif mode == 'ir':
        expected_gt_frames = [round((i / ir_fps) * rgb_fps) for i in range(1601)]

    rgb_gt_by_frame_num = {}
    for _, row in df_gt.iterrows():
        frame_num = row['frame']
        if frame_num not in rgb_gt_by_frame_num:
            rgb_gt_by_frame_num[frame_num] = []

        item_id = row['id']
        if isinstance(item_id, str) or not np.isnan(item_id):
            rgb_gt_by_frame_num[frame_num].append({
                'cx': (row['gt_left'] + row['gt_right']) / 2,
                'cy': (row['gt_top'] + row['gt_bottom']) / 2,
                'w': row['gt_right'] - row['gt_left'],
                'h': row['gt_bottom'] - row['gt_top']
            })

    match_threshold = 50.0 if mode == 'ir' else 50.0

    hit_widths, miss_widths, detection_confs_and_widths = [], [], []
    total_tp, total_fp = 0, 0
    matched_expected_gt_frames, claimed_gts_global = set(), set()
    
    print(f"Scoring all {len(det_df)} detections row-by-row...")
    
    for _, row in det_df.iterrows():
        csv_frame = int(row['frame'])
        
        # Framerate matching
        target_gt_frame = round((csv_frame / ir_fps) * rgb_fps) if mode == 'ir' else csv_frame
            
        det_cx = (row['x1'] + row['x2']) / 2
        det_cy = (row['y1'] + row['y2']) / 2
        det_w = row['x2'] - row['x1']
        det_h = row['y2'] - row['y1']
        conf = row['confidence']
        
        best_dist = match_threshold
        best_gt, best_gt_tuple = None, None
        best_gt_w = 0
        
        if rgb_gt_by_frame_num:
            closest_gt_frame = min(rgb_gt_by_frame_num.keys(), key=lambda f: abs(f - target_gt_frame))
            if abs(closest_gt_frame - target_gt_frame) <= 2:
                for i, gt in enumerate(rgb_gt_by_frame_num[closest_gt_frame]):
                    
                    if mode == 'ir':
                        # Revert to strict geometric resolution scaling (no center-bias cheat)
                        gt_cx = gt['cx'] * (640.0 / 1440.0)
                        gt_cy = gt['cy'] * (512.0 / 1080.0) + 95.0
                        gt_w = gt['w'] * (640.0 / 1440.0)
                        gt_h = gt['h'] * (512.0 / 1080.0)
                    else:
                        gt_cx, gt_cy = gt['cx'], gt['cy']
                        gt_w, gt_h = gt['w'], gt['h']
                    
                    dist = np.sqrt((det_cx - gt_cx)**2 + (det_cy - gt_cy)**2)
                    width_ratio = det_w / max(1, gt_w)
                    height_ratio = det_h / max(1, gt_h)
                    
                    is_right_size = (0.2 < width_ratio < 5.0) and (0.2 < height_ratio < 5.0)
                    
                    if dist < best_dist and is_right_size:
                        if (closest_gt_frame, i) not in claimed_gts_global:
                            best_dist = dist
                            best_gt = gt
                            best_gt_w = gt_w
                            best_gt_tuple = (closest_gt_frame, i)
                        
        if best_gt is not None:
            total_tp += 1
            claimed_gts_global.add(best_gt_tuple)
            detection_confs_and_widths.append((conf, best_gt_w, True))
            
            closest_expected = min(expected_gt_frames, key=lambda f: abs(f - target_gt_frame))
            if abs(closest_expected - target_gt_frame) <= 2:
                matched_expected_gt_frames.add(closest_expected)
                hit_widths.append(best_gt_w)
        else:
            total_fp += 1
            detection_confs_and_widths.append((conf, det_w, False))

    total_fn = 0
    for exp_frame in expected_gt_frames:
        if exp_frame not in matched_expected_gt_frames:
            closest_gt_frame = min(rgb_gt_by_frame_num.keys(), key=lambda f: abs(f - exp_frame))
            if abs(closest_gt_frame - exp_frame) <= 2:
                gt_list = rgb_gt_by_frame_num[closest_gt_frame]
                total_fn += len(gt_list)
                for gt in gt_list:
                    scale_miss = 1.44 if mode == 'ir' else 1.0
                    miss_widths.append(gt['w'] * scale_miss)

    total_frames = len(expected_gt_frames)
    detection_rate = total_tp / max(1, (total_tp + total_fn))
    fppi = total_fp / max(1, total_frames)

    print(f"\n--- YOLO {mode.upper()} CSV EVALUATION RESULTS ---")
    print(f"Total Evaluated Frames: {total_frames}")
    print(f"True Positives (Hits): {total_tp}")
    print(f"False Positives (Alarms): {total_fp}")
    print(f"False Negatives (Misses): {total_fn}")
    print(f"Empirical Detection Rate (EDR): {detection_rate:.4f} ({detection_rate*100:.1f}%)")
    print(f"False Positives Per Image (FPPI): {fppi:.5f}")

    # Keeps your existing plotting functionality completely intact
    save_detection_plots(output_dir, hit_widths, miss_widths, detection_confs_and_widths, prefix=f"yolo_csv_{mode}")
    
    # Return metrics so the comparison plot can use them
    return {'edr': detection_rate, 'fppi': fppi}

def evaluate_ir_classical(output_dir, ir_video_path, ir_fps=9.0, rgb_fps=30.0):
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
    out_video = cv2.VideoWriter(os.path.join(output_dir,"ir_classical_output.mp4"), fourcc, ir_fps, (ir_w, ir_h), isColor=True)

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
        debug_path = os.path.join(output_dir, f"ir_debug_{ir_frame_idx:04d}.png") if ir_frame_idx in [100, 500, 1000] else None
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

        # --- Metrics & Plot Data Collection ---
        for det in confirmed_detections:
            best_dist = 120.0  # Adjust this depending on the function (e.g., 40.0 for IR, 100.0 for RGB)
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_items):
                if i in matched_gt_indices:
                    continue
                
                # Safely handle both dicts (CSV mode) and objects (Fused/Classical mode)
                gt_cx = gt['cx'] if isinstance(gt, dict) else gt.cx
                gt_cy = gt['cy'] if isinstance(gt, dict) else gt.cy
                gt_w = gt['w'] if isinstance(gt, dict) else gt.w
                gt_h = gt['h'] if isinstance(gt, dict) else gt.h
                    
                dist = np.sqrt((det['cx'] - gt_cx)**2 + (det['cy'] - gt_cy)**2)
                
                # Calculate how much bigger or smaller the detection is compared to GT
                width_ratio = det['w'] / max(1, gt_w)
                height_ratio = det['h'] / max(1, gt_h)
                
                # Only count it as a match IF it's close AND the size is within your custom 20% to 200% range
                is_right_size = (0.2 < width_ratio < 2.0) and (0.2 < height_ratio < 2.0)

                if dist < best_dist and is_right_size:
                    best_dist = dist
                    best_gt_idx = i
            
            # Extract confidence score safely
            conf_val = det.get('conf', 0.0) if 'conf' in det else det['conf']
            
            if best_gt_idx != -1:
                # --- CORRECT DETECTION (True Positive) ---
                matched_gt_indices.add(best_gt_idx)
                frame_tp += 1
                
                # Save for the scatter plot (Green)
                matched_gt = gt_items[best_gt_idx]
                matched_gt_w = matched_gt['w'] if isinstance(matched_gt, dict) else matched_gt.w
                detection_confs_and_widths.append((conf_val, matched_gt_w, True))
            else:
                # --- WRONG DETECTION (False Positive) ---
                frame_fp += 1 
                
                # Save for the scatter plot (Red) - using the detection's own width since no GT matched
                detection_confs_and_widths.append((conf_val, det['w'], False))

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

    save_detection_plots(output_dir, hit_widths, miss_widths, detection_confs_and_widths, prefix="ir_classical")


def evaluate_on_ir(output_dir, experiment_name, tracking_weights, transform_weights, ir_video_path, rgb_fps=30.0, ir_fps=9.0):
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
    out_video = cv2.VideoWriter(os.path.join(output_dir,"ir_eval_output.mp4"), fourcc, ir_fps, (ir_w, ir_h), isColor=True)

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

        # --- Metrics & Plot Data Collection ---
        for det in confirmed_detections:
            best_dist = 100.0  # Max distance for IR
            best_gt_idx = -1
            
            # THE FIX: Change gt_items to gt_items_ir here
            for i, gt in enumerate(gt_items_ir):
                if i in matched_gt_indices:
                    continue
                
                # Safely handle both dicts and objects
                gt_cx = gt['cx'] if isinstance(gt, dict) else gt.cx
                gt_cy = gt['cy'] if isinstance(gt, dict) else gt.cy
                gt_w = gt['w'] if isinstance(gt, dict) else gt.w
                gt_h = gt['h'] if isinstance(gt, dict) else gt.h
                    
                dist = np.sqrt((det['cx'] - gt_cx)**2 + (det['cy'] - gt_cy)**2)
                
                # Calculate how much bigger or smaller the detection is compared to GT
                width_ratio = det['w'] / max(1, gt_w)
                height_ratio = det['h'] / max(1, gt_h)
                
                # Only count it as a match IF it's close AND the size is within your 20% to 200% range
                is_right_size = (0.2 < width_ratio < 4.0) and (0.2 < height_ratio < 4.0)

                if dist < best_dist and is_right_size:
                    best_dist = dist
                    best_gt_idx = i
            
            # Extract confidence score safely
            conf_val = det.get('conf', 0.0) if 'conf' in det else det['conf']
            
            if best_gt_idx != -1:
                # --- CORRECT DETECTION (True Positive) ---
                matched_gt_indices.add(best_gt_idx)
                frame_tp += 1
                
                # THE FIX: Change gt_items to gt_items_ir here too
                matched_gt = gt_items_ir[best_gt_idx]
                matched_gt_w = matched_gt['w'] if isinstance(matched_gt, dict) else matched_gt.w
                detection_confs_and_widths.append((conf_val, matched_gt_w, True))
            else:
                # --- WRONG DETECTION (False Positive) ---
                frame_fp += 1 
                
                # Save for the scatter plot (Red)
                detection_confs_and_widths.append((conf_val, det['w'], False))

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

    save_detection_plots(output_dir, hit_widths, miss_widths, detection_confs_and_widths, prefix="ir")


def evaluate_fused(output_dir, rgb_video_path, ir_video_path, fused_weights, conf_threshold=0.15,
                   rgb_fps=30.0, ir_fps=8.57):
    import sys
    import os
    import torch
    import cv2
    import numpy as np
    from tqdm import tqdm
    import common_utils
    import dataset_tracking
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from seg_tracker.seg_tracker import SegDetector
    
    # --- Load fused model ---
    print("Loading fused AI model (RGB+IR)...")
    detector = SegDetector()
    ckpt = torch.load(fused_weights, weights_only=False)
    detector.model_seg.load_state_dict(ckpt["model_state_dict"])
    detector.model_seg.eval()
    print(f"  Loaded fused checkpoint: {fused_weights}")

    # --- Load RGB GT dataset ---
    experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")
    cfg = common_utils.load_config_data("120_hrnet32_all", experiments_dir=experiments_dir)

    val_dataset = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_VALID,
        cfg_data=cfg,
        return_torch_tensors=False,
        small_subset=False
    )

    rgb_gt_by_frame_num = {}
    for i in range(len(val_dataset)):
        frame = val_dataset.frames[i]
        rgb_gt_by_frame_num[frame.frame_num] = frame.items

    # --- Open videos ---
    cap_rgb = cv2.VideoCapture(rgb_video_path)
    cap_ir = cv2.VideoCapture(ir_video_path)

    if not cap_rgb.isOpened() or not cap_ir.isOpened():
        print("Error: cannot open video(s)")
        return

    orig_w = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_COUNT))
    n_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    ir_native_w = int(cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH))
    ir_native_h = int(cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT))

    MODEL_W = 2560  # adjust if your model uses different padding
    MODEL_H = 2048
    x0 = (MODEL_W - orig_w) // 2
    y0 = (MODEL_H - orig_h) // 2

    print(f"  RGB: {orig_w}x{orig_h} @ {rgb_fps} fps ({n_rgb} frames)")
    print(f"  IR:  {ir_native_w}x{ir_native_h} @ {ir_fps} fps ({n_ir} frames)")

    # Side-by-side output
    ir_disp_w = round(ir_native_w * orig_h / ir_native_h)
    out_w = orig_w + ir_disp_w
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(os.path.join(output_dir,"fused_eval_output.mp4"), fourcc, ir_fps, (out_w, orig_h))

    tracker = DelayTracker(min_steps=3, max_distance=50.0, max_coast_frames=5)

    def pad_to_model(img):
        canvas = np.zeros((MODEL_H, MODEL_W), dtype=np.uint8)
        canvas[y0:y0 + orig_h, x0:x0 + orig_w] = img
        return canvas

    prev_rgb_pad = None
    prev_ir_pad = None

    hit_widths = []
    miss_widths = []
    detection_confs_and_widths = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_frames = 0

    print("Running fused evaluation...")

    # --- THE FIX: Track the exact RGB frame manually ---
    current_rgb_idx = -1
    last_valid_rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    cap_ir.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for ir_idx in tqdm(range(n_ir)):
        ret_ir, frame_ir = cap_ir.read()
        if not ret_ir:
            break

        # Time-match RGB to IR mathematically
        rgb_idx = round((ir_idx / ir_fps) * rgb_fps)
        rgb_idx = max(0, min(rgb_idx, n_rgb - 1))

        # --- THE FIX: Sequential catch-up loop ---
        # Physically read frames until we hit the exact target index
        while current_rgb_idx < rgb_idx:
            ret_rgb, temp_frame = cap_rgb.read()
            if not ret_rgb:
                break
            last_valid_rgb = temp_frame
            current_rgb_idx += 1
            
        frame_bgr = last_valid_rgb.copy()

        # --- Proceed with the rest of your logic ---
        gray_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_ir = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
        if gray_ir.shape != (orig_h, orig_w):
            gray_ir = cv2.resize(gray_ir, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        cur_rgb_pad = pad_to_model(gray_rgb)
        cur_ir_pad = pad_to_model(gray_ir)

        if prev_rgb_pad is None:
            prev_rgb_pad = cur_rgb_pad.copy()
            prev_ir_pad = cur_ir_pad.copy()
            continue

        # --- Run fused model (4-channel: prev_RGB, prev_IR, cur_RGB, cur_IR) ---
        h_pad, w_pad = MODEL_H, MODEL_W
        padding = 64

        X = np.zeros((1, 4, h_pad, w_pad + padding * 2), dtype=np.uint8)

        prev_tr = detector.estimate_transformation_full(prev_img=prev_rgb_pad, img=cur_rgb_pad)
        prev_rgb_aligned = cv2.warpAffine(prev_rgb_pad, prev_tr[:2, :],
                                           dsize=(w_pad, h_pad), flags=cv2.INTER_LINEAR)
        prev_ir_aligned = cv2.warpAffine(prev_ir_pad, prev_tr[:2, :],
                                          dsize=(w_pad, h_pad), flags=cv2.INTER_LINEAR)

        X[0, 0, :, padding:-padding] = prev_rgb_aligned  # prev RGB
        X[0, 1, :, padding:-padding] = cur_rgb_pad        # current RGB
        X[0, 2, :, padding:-padding] = prev_ir_aligned    # prev IR
        X[0, 3, :, padding:-padding] = cur_ir_pad         # current IR

        from predict_ensemble import predict_ensemble
        detected = predict_ensemble(
            X=X,
            models_full_res=[detector.model_seg],
            models_crops=[],
            full_res_threshold=conf_threshold,
            x_offset=-padding
        )

        # Filter and convert to tracker format
        raw_detections = []
        fused_dy_offset = -60.0  # shift detections up to match RGB GT
        fused_dx_offset = 0.0
        for det in detected:
            det['cx'] += fused_dx_offset
            det['cy'] += fused_dy_offset
            cx = det['cx'] + det.get('offset', (0, 0))[0] - x0
            cy = det['cy'] + det.get('offset', (0, 0))[1] - y0
            if cx < 0 or cx > orig_w or cy < 0 or cy > orig_h:
                continue
            raw_detections.append({
                'cx': cx,
                'cy': cy,
                'w': det['w'],
                'h': det['h'],
                'conf': det['conf'],
                'tracking': (0.0, 0.0)
            })
        confirmed_detections = tracker.update(raw_detections)

        # --- GT lookup ---
        rgb_frame_num = rgb_idx  # video frame index = frame number
        gt_items = []
        if rgb_gt_by_frame_num:
            closest_frame = min(rgb_gt_by_frame_num.keys(), key=lambda f: abs(f - rgb_frame_num))
            if abs(closest_frame - rgb_frame_num) <= 2:
                gt_items = rgb_gt_by_frame_num[closest_frame]

        if ir_idx % 100 == 0 and raw_detections and gt_items:
            det = raw_detections[0]
            gt = gt_items[0]
            print(f"  Offset: dx={det['cx']-gt.cx:.0f} dy={det['cy']-gt.cy:.0f}")

        # --- Metrics ---
        matched_gt_indices = set()
        frame_tp = 0
        frame_fp = 0

        # --- Metrics & Plot Data Collection ---
        for det in confirmed_detections:
            best_dist = 100.0  # Adjust this depending on the function (e.g., 40.0 for IR, 100.0 for RGB)
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_items):
                if i in matched_gt_indices:
                    continue
                
                # Safely handle both dicts (CSV mode) and objects (Fused/Classical mode)
                gt_cx = gt['cx'] if isinstance(gt, dict) else gt.cx
                gt_cy = gt['cy'] if isinstance(gt, dict) else gt.cy
                gt_w = gt['w'] if isinstance(gt, dict) else gt.w
                gt_h = gt['h'] if isinstance(gt, dict) else gt.h
                    
                dist = np.sqrt((det['cx'] - gt_cx)**2 + (det['cy'] - gt_cy)**2)
                
                # Calculate how much bigger or smaller the detection is compared to GT
                width_ratio = det['w'] / max(1, gt_w)
                height_ratio = det['h'] / max(1, gt_h)
                
                # Only count it as a match IF it's close AND the size is within your custom 20% to 200% range
                is_right_size = (0.2 < width_ratio < 5.0) and (0.2 < height_ratio < 5.0)

                if dist < best_dist and is_right_size:
                    best_dist = dist
                    best_gt_idx = i
            
            # Extract confidence score safely
            conf_val = det.get('conf', 0.0) if 'conf' in det else det['conf']
            
            if best_gt_idx != -1:
                # --- CORRECT DETECTION (True Positive) ---
                matched_gt_indices.add(best_gt_idx)
                frame_tp += 1
                
                # Save for the scatter plot (Green)
                matched_gt = gt_items[best_gt_idx]
                matched_gt_w = matched_gt['w'] if isinstance(matched_gt, dict) else matched_gt.w
                detection_confs_and_widths.append((conf_val, matched_gt_w, True))
            else:
                # --- WRONG DETECTION (False Positive) ---
                frame_fp += 1 
                
                # Save for the scatter plot (Red) - using the detection's own width since no GT matched
                detection_confs_and_widths.append((conf_val, det['w'], False))


        frame_fn = len(gt_items) - len(matched_gt_indices)
        total_tp += frame_tp
        total_fp += frame_fp
        total_fn += frame_fn
        total_frames += 1

        for i, gt in enumerate(gt_items):
            if i in matched_gt_indices:
                hit_widths.append(gt.w)
            else:
                miss_widths.append(gt.w)


        # --- Diagnostic ---
        if ir_idx % 100 == 0:
            gt_str = ""
            if gt_items and confirmed_detections:
                gt = gt_items[0]
                det = confirmed_detections[0]
                dist = np.sqrt((det['cx'] - gt.cx)**2 + (det['cy'] - gt.cy)**2)
                gt_str = f" | GT:({gt.cx:.0f},{gt.cy:.0f}) Det:({det['cx']:.0f},{det['cy']:.0f}) Dist:{dist:.0f}px"
            print(f"Frame {ir_idx}: {len(raw_detections)} raw, {len(confirmed_detections)} confirmed, "
                  f"{len(gt_items)} GT{gt_str}")

        # --- Visualization: side-by-side RGB + IR ---
        display_rgb = frame_bgr.copy()

        # GT in red on RGB
        for gt in gt_items:
            x1, y1 = int(gt.cx - gt.w / 2), int(gt.cy - gt.h / 2)
            x2, y2 = int(gt.cx + gt.w / 2), int(gt.cy + gt.h / 2)
            cv2.rectangle(display_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display_rgb, f"GT d={gt.distance:.0f}m", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Detections in green on RGB
        for det in confirmed_detections:
            x1, y1 = int(det['cx'] - det['w'] / 2), int(det['cy'] - det['h'] / 2)
            x2, y2 = int(det['cx'] + det['w'] / 2), int(det['cy'] + det['h'] / 2)
            conf = int(det.get('conf', 0.0) * 100)
            track_id = det.get('track_id', '?')
            cv2.rectangle(display_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_rgb, f"ID:{track_id} {conf}%", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # IR panel — scale detections to IR display space
        ir_display = cv2.resize(frame_ir, (ir_disp_w, orig_h), interpolation=cv2.INTER_LINEAR)
        scale_ir_x = ir_disp_w / orig_w
        for det in confirmed_detections:
            cx_ir = round(det['cx'] * scale_ir_x)
            dw_ir = round(det['w'] * scale_ir_x)
            x1_ir, y1_ir = int(cx_ir - dw_ir / 2), int(det['cy'] - det['h'] / 2)
            x2_ir, y2_ir = int(cx_ir + dw_ir / 2), int(det['cy'] + det['h'] / 2)
            cv2.rectangle(ir_display, (x1_ir, y1_ir), (x2_ir, y2_ir), (0, 255, 0), 2)

        cv2.putText(display_rgb, f"RGB f{rgb_idx} | Fused conf>{conf_threshold}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(ir_display, "IR", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        composite = np.concatenate([display_rgb, ir_display], axis=1)
        out_video.write(composite)

        prev_rgb_pad = cur_rgb_pad.copy()
        prev_ir_pad = cur_ir_pad.copy()

    cap_rgb.release()
    cap_ir.release()
    out_video.release()

    detection_rate = total_tp / max(1, (total_tp + total_fn))
    fppi = total_fp / max(1, total_frames)

    print(f"\n--- FUSED EVALUATION RESULTS ---")
    print(f"Total Frames Evaluated: {total_frames}")
    print(f"True Positives (Hits): {total_tp}")
    print(f"False Positives (Alarms): {total_fp}")
    print(f"False Negatives (Misses): {total_fn}")
    print(f"Empirical Detection Rate (EDR): {detection_rate:.4f} ({detection_rate*100:.1f}%)")
    print(f"False Positives Per Image (FPPI): {fppi:.5f}")
    print("Video saved to fused_eval_output.mp4")

    save_detection_plots(output_dir, hit_widths, miss_widths, detection_confs_and_widths, prefix="fused")

def evaluate_rgb_at_ir_fps(experiment_name, tracking_weights, model_weights_path, 
                            transform_weights, rgb_video_path, output_dir="eval_results",
                            rgb_fps=30.0, ir_fps=9.0):
    """RGB-only evaluation but only at IR frame times, for fair comparison with fused."""
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

    val_dataset = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_VALID,
        cfg_data=cfg,
        return_torch_tensors=False,
        small_subset=False
    )

    rgb_gt_by_frame_num = {}
    for i in range(len(val_dataset)):
        frame = val_dataset.frames[i]
        rgb_gt_by_frame_num[frame.frame_num] = frame.items

    cap_rgb = cv2.VideoCapture(rgb_video_path)
    orig_w = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    n_ir = int(n_rgb * ir_fps / rgb_fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(os.path.join(output_dir, "rgb_at_ir_fps_eval.mp4"),
                                 fourcc, ir_fps, (orig_w, orig_h), isColor=True)

    tracker = DelayTracker(min_steps=5, max_distance=40.0, max_coast_frames=3)

    hit_widths = []
    miss_widths = []
    detection_confs_and_widths = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_frames = 0

    prev_frame = None

    print(f"Running RGB-only at IR fps ({ir_fps} fps, {n_ir} frames)...")

    for ir_idx in tqdm(range(n_ir)):
        rgb_idx = round((ir_idx / ir_fps) * rgb_fps)
        rgb_idx = max(0, min(rgb_idx, n_rgb - 1))

        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, rgb_idx)
        ret, frame_bgr = cap_rgb.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if prev_frame is None:
            prev_frame = frame_gray
            continue

        prev_aligned = align_frames_dl(transform_model, frame_gray, prev_frame)
        tensor_inputs = prepare_inputs(frame_gray, prev_aligned)
        with torch.amp.autocast('cuda'), torch.no_grad():
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

        # Rescale from model space to original
        scale_x = orig_w / 2432.0
        scale_y = orig_h / 2048.0
        for det in raw_detections:
            det['cx'] *= scale_x
            det['cy'] *= scale_y
            det['w'] *= scale_x
            det['h'] *= scale_y
            det['tracking'] = (det['tracking'][0] * scale_x, det['tracking'][1] * scale_y)

        confirmed_detections = tracker.update(raw_detections)

        # GT lookup
        rgb_frame_num = rgb_idx
        gt_items = []
        if rgb_gt_by_frame_num:
            closest_frame = min(rgb_gt_by_frame_num.keys(), key=lambda f: abs(f - rgb_frame_num))
            if abs(closest_frame - rgb_frame_num) <= 2:
                gt_items = rgb_gt_by_frame_num[closest_frame]

        # Metrics
        matched_gt_indices = set()
        frame_tp = 0
        frame_fp = 0

        # --- Metrics & Plot Data Collection ---
        for det in confirmed_detections:
            best_dist = 40.0  # Adjust this depending on the function (e.g., 40.0 for IR, 100.0 for RGB)
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_items):
                if i in matched_gt_indices:
                    continue
                
                # Safely handle both dicts (CSV mode) and objects (Fused/Classical mode)
                gt_cx = gt['cx'] if isinstance(gt, dict) else gt.cx
                gt_cy = gt['cy'] if isinstance(gt, dict) else gt.cy
                gt_w = gt['w'] if isinstance(gt, dict) else gt.w
                gt_h = gt['h'] if isinstance(gt, dict) else gt.h
                    
                dist = np.sqrt((det['cx'] - gt_cx)**2 + (det['cy'] - gt_cy)**2)
                
                # Calculate how much bigger or smaller the detection is compared to GT
                width_ratio = det['w'] / max(1, gt_w)
                height_ratio = det['h'] / max(1, gt_h)
                
                # Only count it as a match IF it's close AND the size is within your custom 20% to 200% range
                is_right_size = (0.2 < width_ratio < 2.0) and (0.2 < height_ratio < 2.0)

                if dist < best_dist and is_right_size:
                    best_dist = dist
                    best_gt_idx = i
            
            # Extract confidence score safely
            conf_val = det.get('conf', 0.0) if 'conf' in det else det['conf']
            
            if best_gt_idx != -1:
                # --- CORRECT DETECTION (True Positive) ---
                matched_gt_indices.add(best_gt_idx)
                frame_tp += 1
                
                # Save for the scatter plot (Green)
                matched_gt = gt_items[best_gt_idx]
                matched_gt_w = matched_gt['w'] if isinstance(matched_gt, dict) else matched_gt.w
                detection_confs_and_widths.append((conf_val, matched_gt_w, True))
            else:
                # --- WRONG DETECTION (False Positive) ---
                frame_fp += 1 
                
                # Save for the scatter plot (Red) - using the detection's own width since no GT matched
                detection_confs_and_widths.append((conf_val, det['w'], False))

        frame_fn = len(gt_items) - len(matched_gt_indices)
        total_tp += frame_tp
        total_fp += frame_fp
        total_fn += frame_fn
        total_frames += 1

        for i, gt in enumerate(gt_items):
            if i in matched_gt_indices:
                hit_widths.append(gt.w)
            else:
                miss_widths.append(gt.w)

        # Visualization
        display_frame = frame_bgr.copy()
        for gt in gt_items:
            x1, y1 = int(gt.cx - gt.w / 2), int(gt.cy - gt.h / 2)
            x2, y2 = int(gt.cx + gt.w / 2), int(gt.cy + gt.h / 2)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for det in confirmed_detections:
            x1, y1 = int(det['cx'] - det['w'] / 2), int(det['cy'] - det['h'] / 2)
            x2, y2 = int(det['cx'] + det['w'] / 2), int(det['cy'] + det['h'] / 2)
            conf = int(det.get('conf', 0.0) * 100)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{conf}%", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out_video.write(display_frame)

        if ir_idx % 100 == 0:
            print(f"Frame {ir_idx}: {len(raw_detections)} raw, {len(confirmed_detections)} confirmed, {len(gt_items)} GT")

        prev_frame = frame_gray

    cap_rgb.release()
    out_video.release()

    detection_rate = total_tp / max(1, (total_tp + total_fn))
    fppi = total_fp / max(1, total_frames)

    print(f"\n--- RGB-ONLY @ IR FPS EVALUATION RESULTS ---")
    print(f"Total Frames Evaluated: {total_frames}")
    print(f"True Positives (Hits): {total_tp}")
    print(f"False Positives (Alarms): {total_fp}")
    print(f"False Negatives (Misses): {total_fn}")
    print(f"Empirical Detection Rate (EDR): {detection_rate:.4f} ({detection_rate*100:.1f}%)")
    print(f"False Positives Per Image (FPPI): {fppi:.5f}")
    print(f"Video saved to {os.path.join(output_dir, 'rgb_at_ir_fps_eval.mp4')}")

    save_detection_plots(output_dir, hit_widths, miss_widths, detection_confs_and_widths, prefix="rgb_at_ir_fps")

def save_detection_plots(output_dir, hit_widths, miss_widths, detection_confs_and_widths, prefix="rgb"):
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
    plt.savefig(os.path.join(output_dir,f'{prefix}_detection_by_size.png'), dpi=150)
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
    plt.savefig(os.path.join(output_dir,f'{prefix}_detection_rate_by_size.png'), dpi=150)
    plt.close()
    print(f"Plot saved to {prefix}_detection_rate_by_size.png")

    # 3. Confidence vs size with trendlines
    if detection_confs_and_widths:
        # Unpack the new 3-element tuples
        confs, widths, is_correct = zip(*detection_confs_and_widths)
        confs = np.array(confs, dtype=np.float64)
        widths = np.array(widths, dtype=np.float64)
        is_correct = np.array(is_correct, dtype=bool)

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        
        # --- Masks ---
        correct_mask = is_correct
        wrong_mask = ~is_correct

        # --- Plot Scatter Points ---
        ax3.scatter(widths[correct_mask], confs[correct_mask], alpha=0.4, s=15, color='green', label=f'Correct ({np.sum(correct_mask)})')
        ax3.scatter(widths[wrong_mask], confs[wrong_mask], alpha=0.4, s=15, color='red', label=f'Wrong ({np.sum(wrong_mask)})')

        # --- Trendline: Correct Detections (Green) ---
        if np.sum(correct_mask) > 3:
            w_corr = widths[correct_mask]
            c_corr = confs[correct_mask]
            
            sort_idx = np.argsort(w_corr)
            w_corr_sorted = w_corr[sort_idx]
            c_corr_sorted = c_corr[sort_idx]
            
            # Need enough unique x-values to fit a 3rd order polynomial securely
            if len(np.unique(w_corr_sorted)) > 3:
                coeffs_corr = np.polyfit(w_corr_sorted, c_corr_sorted, 3)
                trend_x_corr = np.linspace(w_corr_sorted.min(), w_corr_sorted.max(), 200)
                trend_y_corr = np.clip(np.polyval(coeffs_corr, trend_x_corr), 0, 1)
                ax3.plot(trend_x_corr, trend_y_corr, color='darkgreen', linewidth=2.5, label='Correct Detections Trend')

        # --- Trendline: Wrong Detections (Red) ---
        if np.sum(wrong_mask) > 3:
            w_wrong = widths[wrong_mask]
            c_wrong = confs[wrong_mask]
            
            sort_idx = np.argsort(w_wrong)
            w_wrong_sorted = w_wrong[sort_idx]
            c_wrong_sorted = c_wrong[sort_idx]
            
            if len(np.unique(w_wrong_sorted)) > 3:
                coeffs_wrong = np.polyfit(w_wrong_sorted, c_wrong_sorted, 3)
                trend_x_wrong = np.linspace(w_wrong_sorted.min(), w_wrong_sorted.max(), 200)
                trend_y_wrong = np.clip(np.polyval(coeffs_wrong, trend_x_wrong), 0, 1)
                ax3.plot(trend_x_wrong, trend_y_wrong, color='darkred', linewidth=2.5, label='False Detections Trend')

        ax3.set_xlabel('Bounding Box Width (px)')
        ax3.set_ylabel('Confidence Score')
        ax3.set_ylim(0, 1.05)
        ax3.set_title(f'[{prefix.upper()}] Detection Confidence vs Object Size')
        ax3.legend()
        plt.grid(True, linestyle='--', alpha=0.25)
        
        plt.savefig(os.path.join(output_dir, f'{prefix}_confidence_vs_size.png'), dpi=150)
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

def run_inference(output_dir, experiment_name, model_weights_path, video_path, output_path="output_tracked.mp4"):
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
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="visualize",
                        choices=["visualize", "evaluate_rgb", "evaluate_ir",
                                 "evaluate_ir_classical", "evaluate_fused",
                                 "evaluate_rgb_baseline",
                                 "yolo_rgb", "yolo_ir", "yolo_rgbt",
                                 "yolo_csv_rgb", "yolo_csv_ir", "yolo_csv_rgbt",
                                 "yolo_csv_compare"]) # <-- Added new mode here
    parser.add_argument("--det_csv", type=str, help="Path to YOLO detection CSV")
    
    # <-- Added specific CSV arguments for the comparison mode -->
    parser.add_argument("--rgb_csv", type=str, help="Path to RGB CSV for comparison")
    parser.add_argument("--ir_csv", type=str, help="Path to IR CSV for comparison")
    parser.add_argument("--fused_csv", type=str, help="Path to Fused CSV for comparison")
    
    parser.add_argument("--yolo_weights", type=str, default=None,
                        help="Path to YOLO .pt weights")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--ir_video", type=str, help="Path to IR video for evaluate_ir mode")
    parser.add_argument("--exp", type=str, default="my_hrnet_experiment", help="Experiment name")
    parser.add_argument("--model_weights", type=str,
                        default="/cluster/home/nbaruffol/airborne_detection/output/checkpoints/120_hrnet32_all/0/2220.pt")
    parser.add_argument("--tracking_weights", type=str,
                        default="/cluster/home/nbaruffol/airborne_detection/output/checkpoints/120_hrnet32_all/0/2220.pt")
    parser.add_argument("--transform_weights", type=str,
                        default="/cluster/home/nbaruffol/airborne_detection/output/checkpoints/030_tr_tsn_rn34_w3_crop_borders/0/500.pt")
    parser.add_argument("--fused_weights", type=str,
                        default="/cluster/home/nbaruffol/airborne_detection/output/models/120_hrnet32_fused_2220.pth",
                        help="Path to fused model checkpoint")
    parser.add_argument("--conf_threshold", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Directory for all output files")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)


    if args.mode == "visualize":
        run_inference(
            output_dir=args.output_dir,
            experiment_name=args.exp,
            model_weights_path=args.model_weights,
            video_path=args.video,
            output_path=os.path.join(args.output_dir, "rgb_visualize.mp4")
        )
    elif args.mode == "evaluate_rgb":
        evaluate_tracker(
            experiment_name=args.exp,
            tracking_weights=args.tracking_weights,
            model_weights_path=args.model_weights,
            transform_weights=args.transform_weights,
            output_dir=args.output_dir,
        )
    elif args.mode == "evaluate_ir":
        evaluate_on_ir(
            experiment_name=args.exp,
            tracking_weights=args.tracking_weights,
            transform_weights=args.transform_weights,
            ir_video_path=args.ir_video,
            output_dir=args.output_dir,
        )
    elif args.mode == "evaluate_ir_classical":
        evaluate_ir_classical(
            ir_video_path=args.ir_video,
            output_dir=args.output_dir,
        )
    elif args.mode == "evaluate_fused":
        evaluate_fused(
            rgb_video_path=args.video,
            ir_video_path=args.ir_video,
            fused_weights=args.fused_weights,
            conf_threshold=args.conf_threshold,
            output_dir=args.output_dir,
        )
    elif args.mode == "evaluate_rgb_baseline":
        evaluate_rgb_at_ir_fps(
            experiment_name=args.exp,
            tracking_weights=args.tracking_weights,
            model_weights_path=args.model_weights,
            transform_weights=args.transform_weights,
            rgb_video_path=args.video,
            output_dir=args.output_dir,
        )
    elif args.mode == "yolo_rgb":
        evaluate_yolo(
            yolo_weights=args.yolo_weights,
            mode="rgb",
            rgb_video_path=args.video,
            output_dir=args.output_dir,
        )
    elif args.mode == "yolo_ir":
        evaluate_yolo(
            yolo_weights=args.yolo_weights,
            mode="ir",
            ir_video_path=args.ir_video,
            output_dir=args.output_dir,
        )
    elif args.mode == "yolo_rgbt":
        evaluate_yolo(
            yolo_weights=args.yolo_weights,
            mode="rgbt",
            rgb_video_path=args.video,
            ir_video_path=args.ir_video,
            output_dir=args.output_dir,
        )
    elif args.mode == "yolo_csv_rgb":
        evaluate_yolo_csv(csv_path=args.det_csv, mode="rgb", output_dir=args.output_dir)
    elif args.mode == "yolo_csv_ir":
        evaluate_yolo_csv(csv_path=args.det_csv, mode="ir", output_dir=args.output_dir)
    elif args.mode == "yolo_csv_rgbt":
        evaluate_yolo_csv(csv_path=args.det_csv, mode="rgbt", output_dir=args.output_dir)
        
    # <-- NEW COMPARISON BLOCK -->
    elif args.mode == "yolo_csv_compare":
        results = {}
        if args.rgb_csv and os.path.exists(args.rgb_csv):
            results['rgb'] = evaluate_yolo_csv(csv_path=args.rgb_csv, mode="rgb", output_dir=args.output_dir)
        if args.ir_csv and os.path.exists(args.ir_csv):
            results['ir'] = evaluate_yolo_csv(csv_path=args.ir_csv, mode="ir", output_dir=args.output_dir)
        if args.fused_csv and os.path.exists(args.fused_csv):
            results['rgbt'] = evaluate_yolo_csv(csv_path=args.fused_csv, mode="rgbt", output_dir=args.output_dir)
            
        if len(results) > 1:
            plot_comparison(results, output_dir=args.output_dir)
        else:
            print("Notice: Provide at least two valid CSVs using --rgb_csv, --ir_csv, or --fused_csv to generate the comparison plot.")
    print("All done!")