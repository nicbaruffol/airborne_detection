"""
Visualize fused model predictions against ground truth.
Saves one MP4 per flight to ../output/vis/{model}_{epoch}/

Usage:
    python visualize_predictions.py 120_hrnet32_fused --epoch 2220 --conf 0.5 --flights 5
"""
import argparse
import os
import pickle

import cv2
import numpy as np
import pandas as pd

import config
from check_frame_level_prediction import load_flights_for_part

# Colours (BGR)
COL_GT   = (255, 80,  80)   # blue  — ground truth
COL_TP   = (80,  220, 80)   # green — detected (above conf threshold)
COL_FP   = (80,  80,  220)  # red   — false positive


def draw_box(img, cx, cy, w, h, colour, label=None):
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def visualize(experiment_name, epoch, conf_threshold=0.5, nb_flights=5, part='part1'):
    oof_dir = f"../output/oof_cache/{experiment_name}_{epoch}"
    out_dir = f"../output/vis/{experiment_name}_{epoch}"
    os.makedirs(out_dir, exist_ok=True)

    gt_df = pd.read_csv(f'{config.DATA_DIR}/{part}/ImageSets/groundtruth.csv')
    # frame number from img_name: "frame_0042.jpg" -> 42
    gt_df['frame_num'] = gt_df['img_name'].str.extract(r'(\d+)').astype(int)

    flights = load_flights_for_part(part)[:nb_flights]

    for flight in flights:
        flight_id = flight.flight_id
        pkl_path = f"{oof_dir}/{part}_{flight_id}.pkl"
        if not os.path.exists(pkl_path):
            print(f"  no predictions for {flight_id}, skipping")
            continue

        predictions = pickle.load(open(pkl_path, 'rb'))
        gt_flight = gt_df[gt_df['flight_id'] == flight_id]

        # Determine crop offset used during inference (must match predict_oof.py logic)
        # Load one image to get its dimensions
        sample_rgb = cv2.imread(
            flight.file_names[0].replace('.jpg', '_rgb.jpg'), cv2.IMREAD_GRAYSCALE)
        if sample_rgb is None:
            print(f"  cannot read images for {flight_id}, skipping")
            continue
        raw_h, raw_w = sample_rgb.shape
        crop_h = (raw_h // 32) * 32
        crop_w = (raw_w // 32) * 32
        crop_y0 = max((raw_h - crop_h) // 2, 0)
        crop_x0 = max((raw_w - crop_w) // 2, 0)

        out_path = f"{out_dir}/{flight_id}.mp4"
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            10,
            (raw_w, raw_h)
        )

        print(f"  rendering {flight_id}  ({len(flight.frame_numbers)} frames) -> {out_path}")

        for frame_idx, frame_num in enumerate(flight.frame_numbers):
            rgb_path = flight.file_names[frame_idx].replace('.jpg', '_rgb.jpg')
            frame_img = cv2.imread(rgb_path)
            if frame_img is None:
                frame_img = np.zeros((raw_h, raw_w, 3), dtype=np.uint8)

            # Ground truth boxes for this frame
            gt_rows = gt_flight[gt_flight['frame_num'] == frame_num]
            gt_boxes = []
            for _, row in gt_rows.iterrows():
                cx = (row['gt_left'] + row['gt_right']) / 2
                cy = (row['gt_top']  + row['gt_bottom']) / 2
                w  = row['gt_right'] - row['gt_left']
                h  = row['gt_bottom'] - row['gt_top']
                gt_boxes.append((cx, cy, w, h))
                draw_box(frame_img, cx, cy, w, h, COL_GT, 'GT')

            # Predictions for this frame
            if frame_idx < len(predictions):
                frame_preds = predictions[frame_idx]
                for det in frame_preds:
                    if det['conf'] < conf_threshold:
                        continue
                    # Convert from crop coords to full-image coords
                    cx = det['cx'] + crop_x0 + det['offset'][0]
                    cy = det['cy'] + crop_y0 + det['offset'][1]
                    w, h = det['w'], det['h']

                    # Check if it matches any GT box
                    box = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
                    matched = any(
                        iou(box, (gx - gw/2, gy - gh/2, gx + gw/2, gy + gh/2)) > 0.1
                        for gx, gy, gw, gh in gt_boxes
                    )
                    col = COL_TP if matched else COL_FP
                    draw_box(frame_img, cx, cy, w, h, col,
                             f"{det['conf']:.2f}")

            # Frame counter overlay
            cv2.putText(frame_img, f"frame {frame_num}  conf>{conf_threshold}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            writer.write(frame_img)

        writer.release()
        print(f"    saved {out_path}")

    print(f"\nDone. Videos in {out_dir}/")
    print("Legend:  BLUE=GT   GREEN=true positive   RED=false positive")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--epoch',    type=int, default=2220)
    parser.add_argument('--conf',     type=float, default=0.5)
    parser.add_argument('--flights',  type=int, default=5)
    parser.add_argument('--part',     type=str, default='part1')
    args = parser.parse_args()

    visualize(args.experiment, args.epoch, args.conf, args.flights, args.part)
