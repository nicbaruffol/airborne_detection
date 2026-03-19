"""
Run the airborne-object detector on a video.

RGB-only mode (default):
    python run_my_video.py --rgb path/to/video.mp4 --output path/to/out.mp4

Fused RGB+IR mode:
    python run_my_video.py --rgb path/to/rgb.mp4 --ir path/to/ir.mp4 --output path/to/out.mp4

Optional: override the segmentation checkpoint with --ckpt.
"""
import argparse
import cv2
import os
import sys
from tqdm import tqdm
import numpy as np
import torch

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/seg_tracker')

from seg_tracker.seg_tracker import SegDetector, SegTrackerFromOffset, SegTracker

# Model expects this padded resolution
MODEL_W, MODEL_H = 2448, 2048

# Trained fused checkpoint (epoch 2220)
CKPT_FUSED = f'{current_path}/output/checkpoints/120_hrnet32_fused/0/2220.pt'


# ─── RGB-only ──────────────────────────────────────────────────────────────────
def track_custom_video(input_video_path, output_video_path):
    print("Loading AI Model (RGB-only)...")
    detector = SegDetector()
    tracker = SegTrackerFromOffset(detector=detector)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}. Check the path!")
        return

    fps          = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x_offset = (MODEL_W - orig_width)  // 2
    y_offset = (MODEL_H - orig_height) // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_width, orig_height))

    print(f"Processing {total_frames} frames from {input_video_path}...")

    prev_frame = None
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        padded_gray  = np.zeros((MODEL_H, MODEL_W), dtype=np.uint8)
        padded_gray[y_offset:y_offset + orig_height, x_offset:x_offset + orig_width] = gray_frame
        inverted_gray = cv2.bitwise_not(padded_gray)

        if prev_frame is None:
            prev_frame = inverted_gray.copy()

        results = tracker.predict(image=inverted_gray, prev_image=prev_frame)

        if results:
            for res in results:
                if isinstance(res, dict):
                    cx, cy, w, h = res['cx'], res['cy'], res['w'], res['h']
                    track_id = res.get('track_id', -1)
                    conf     = res.get('conf', 1.0)
                else:
                    cx, cy, w, h = res.cx, res.cy, res.w, res.h
                    track_id = res.track_id
                    conf     = res.confidence

                cx -= x_offset
                cy -= y_offset
                if cx < 0 or cx > orig_width or cy < 0 or cy > orig_height:
                    continue

                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id} ({conf:.2f})",
                            (x1, max(y1 - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)
        prev_frame = inverted_gray.copy()

    cap.release()
    out.release()
    print(f"Done! Saved tracked video to {output_video_path}")


# ─── Fused RGB+IR ──────────────────────────────────────────────────────────────
def track_fused_video(rgb_path, ir_path, output_path, seg_ckpt=CKPT_FUSED, conf_threshold=0.35):
    """
    Run the 4-channel fused model (prev_RGB, prev_IR, cur_RGB, cur_IR).

    Only processes RGB frames whose timestamp falls within the IR video's
    duration (frames where a real IR counterpart exists).
    IR frames are time-matched via: ir_idx = round(t_rgb * ir_fps)

    Detection filtering: conf >= conf_threshold only.
    SegTracker's conf>0.75 and distance>800 hard filters are intentionally
    skipped — they suppress all detections on out-of-distribution footage.
    """
    print("Loading fused AI model (RGB+IR)...")
    detector = SegDetector()

    # Replace the init weights with the properly trained checkpoint
    ckpt = torch.load(seg_ckpt)
    detector.model_seg.load_state_dict(ckpt["model_state_dict"])
    detector.model_seg.eval()
    print(f"  Loaded seg checkpoint: {seg_ckpt}")

    cap_rgb = cv2.VideoCapture(rgb_path)
    cap_ir  = cv2.VideoCapture(ir_path)
    if not cap_rgb.isOpened():
        print(f"Error: cannot open RGB video {rgb_path}")
        return
    if not cap_ir.isOpened():
        print(f"Error: cannot open IR video {ir_path}")
        return

    rgb_fps     = cap_rgb.get(cv2.CAP_PROP_FPS)
    ir_fps      = cap_ir.get(cv2.CAP_PROP_FPS)
    n_rgb       = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    n_ir        = int(cap_ir.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w      = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h      = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ir_native_w = int(cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH))
    ir_native_h = int(cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # IR display width scaled to RGB height at native aspect ratio
    ir_disp_w = round(ir_native_w * orig_h / ir_native_h)
    out_w     = orig_w + ir_disp_w

    print(f"  RGB: {orig_w}×{orig_h} @ {rgb_fps:.1f} fps  ({n_rgb} frames, {n_rgb/rgb_fps:.1f}s)")
    print(f"  IR : {ir_native_w}×{ir_native_h} @ {ir_fps:.1f} fps  ({n_ir} frames, {n_ir/ir_fps:.1f}s)")
    print(f"  Output: {n_ir} frames at {ir_fps:.1f} fps — side-by-side {out_w}×{orig_h}")

    # Padding offsets to centre the frame in the model canvas
    x0 = (MODEL_W - orig_w) // 2
    y0 = (MODEL_H - orig_h) // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Output runs at IR fps — side-by-side composite
    writer = cv2.VideoWriter(output_path, fourcc, ir_fps, (out_w, orig_h))

    def read_rgb_at(rgb_idx):
        """Seek to rgb_idx and return the BGR frame."""
        rgb_idx = max(0, min(rgb_idx, n_rgb - 1))
        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, rgb_idx)
        ret, f = cap_rgb.read()
        return f if ret else np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    def pad_to_model(img):
        canvas = np.zeros((MODEL_H, MODEL_W), dtype=np.uint8)
        canvas[y0:y0 + orig_h, x0:x0 + orig_w] = img
        return canvas

    prev_rgb_pad = None
    prev_ir_pad  = None

    cap_ir.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for ir_idx in tqdm(range(n_ir)):
        ret_ir, frame_ir = cap_ir.read()
        if not ret_ir:
            break

        # Time-match: nearest RGB frame for this IR timestamp
        rgb_idx = round((ir_idx / ir_fps) * rgb_fps)
        frame = read_rgb_at(rgb_idx)

        gray_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_ir  = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
        if gray_ir.shape != (orig_h, orig_w):
            gray_ir = cv2.resize(gray_ir, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        cur_rgb_pad = pad_to_model(gray_rgb)
        cur_ir_pad  = pad_to_model(gray_ir)

        if prev_rgb_pad is None:
            prev_rgb_pad = cur_rgb_pad.copy()
            prev_ir_pad  = cur_ir_pad.copy()

        # Call detect_objects directly — bypasses SegTracker's conf>0.75 and
        # distance>800 hard filters that suppress detections on new footage
        detected, _ = detector.detect_objects(
            cur_rgb=cur_rgb_pad, cur_ir=cur_ir_pad,
            prev_rgb=prev_rgb_pad, prev_ir=prev_ir_pad
        )

        # Collect filtered detections in RGB coordinate space
        filtered_dets = []
        for det in detected:
            if det['conf'] < conf_threshold:
                continue
            # offset is sub-pixel refinement; x0/y0 undo the padding
            cx   = det['cx'] + det['offset'][0] - x0
            cy   = det['cy'] + det['offset'][1] - y0
            dw   = det['w']
            dh   = det['h']
            conf = det['conf']
            if cx < 0 or cx > orig_w or cy < 0 or cy > orig_h:
                continue  # hallucination in the black border
            filtered_dets.append((cx, cy, dw, dh, conf))

        # Draw on RGB frame
        for cx, cy, dw, dh, conf in filtered_dets:
            x1, y1 = int(cx - dw / 2), int(cy - dh / 2)
            x2, y2 = int(cx + dw / 2), int(cy + dh / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Scale raw IR to native AR at RGB height for display
        ir_display = cv2.resize(frame_ir, (ir_disp_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Draw detections on IR panel (x scaled from stretched→native AR)
        scale_x = ir_disp_w / orig_w
        for cx, cy, dw, dh, conf in filtered_dets:
            cx_ir = round(cx * scale_x)
            dw_ir = round(dw * scale_x)
            x1_ir, y1_ir = int(cx_ir - dw_ir / 2), int(cy - dh / 2)
            x2_ir, y2_ir = int(cx_ir + dw_ir / 2), int(cy + dh / 2)
            cv2.rectangle(ir_display, (x1_ir, y1_ir), (x2_ir, y2_ir), (0, 255, 0), 2)
            cv2.putText(ir_display, f"{conf:.2f}", (x1_ir, max(y1_ir - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Labels and frame counter
        cv2.putText(frame, f"RGB  f{rgb_idx}  IR#{ir_idx}  conf>{conf_threshold}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(ir_display, "IR", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        composite = np.concatenate([frame, ir_display], axis=1)
        writer.write(composite)
        prev_rgb_pad = cur_rgb_pad.copy()
        prev_ir_pad  = cur_ir_pad.copy()

    cap_rgb.release()
    cap_ir.release()
    writer.release()
    print(f"Done! Saved to {output_path}")


# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Airborne object detector on video')
    parser.add_argument('--rgb',    required=True,      help='Path to RGB video')
    parser.add_argument('--ir',     default=None,       help='Path to IR video → enables fused RGB+IR mode')
    parser.add_argument('--output', required=True,      help='Output video path')
    parser.add_argument('--ckpt',   default=CKPT_FUSED, help='Fused seg-model checkpoint (default: epoch 2220)')
    parser.add_argument('--conf',   default=0.35, type=float, help='Confidence threshold (default: 0.35)')
    args = parser.parse_args()

    if args.ir:
        track_fused_video(args.rgb, args.ir, args.output, seg_ckpt=args.ckpt, conf_threshold=args.conf)
    else:
        track_custom_video(args.rgb, args.output)
