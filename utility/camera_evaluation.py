#!/usr/bin/env python3
"""
Camera Evaluation Tool — Drone Detection & Tracking
====================================================
Reads camera specs from CSV files and generates a self-contained,
offline-capable interactive HTML report.

Usage:
    python3 scripts/camera_evaluation.py
"""

import json
import os
import sys
from pathlib import Path

# Allow imports from scripts/utils/
sys.path.insert(0, str(Path(__file__).parent))

from utils.camera_utils import read_csv, compute_fov, assign_colors, build_camera_json
from utils.html_template import build_html

try:
    from plotly.offline import get_plotlyjs
except ImportError:
    print("ERROR: plotly not installed.  Run: pip install plotly", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "data"
OUTPUT_DIR  = REPO_ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "camera_evaluation.html"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DISTANCES_M = list(range(1, 1001))          # 1 … 1000 m, step 1 m

DRONE_DEFAULTS = {
    "mini": {
        "label":  "Mini (DJI Mini 4 Pro)",
        "size_m": 0.300,
        "note":   "Unfolded width with propellers 362 mm — DJI official spec",
    },
    "medium": {
        "label":  "Medium (DJI Mavic 4 Pro)",
        "size_m": 0.400,
        "note":   "Unfolded width 329 mm — DJI official spec",
    },
    "industrial": {
        "label":  "Industrial (DJI Inspire 3)",
        "size_m": 0.500,
        "note":   "Motor-to-motor diagonal 695 mm — DJI official spec",
    },
}

THRESHOLD_DEFAULTS = {
    "detect": 40,   # pixels — typical YOLO / detector minimum
    "track":  10,   # pixels — typical tracker minimum for reliable ID
}

# Blue palette for RGB cameras
RGB_PALETTE = [
    "#1f77b4", "#4a90d9", "#5ba3e0", "#2166ac",
    "#4393c3", "#74add1", "#abd9e9", "#08519c",
    "#2171b5", "#6baed6", "#c6dbef",
]

# Red/orange palette for thermal cameras
THERMAL_PALETTE = [
    "#d73027", "#f46d43", "#fdae61", "#b2182b",
    "#ef6548", "#fc8d59", "#fdd49e", "#a50026",
    "#d6604d", "#f4a582", "#fddbc7",
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load camera data
    print(f"Reading RGB cameras …", flush=True)
    rgb = read_csv(DATA_DIR / "rgb_cameras.csv")
    print(f"  {len(rgb)} cameras loaded.")

    print(f"Reading Thermal cameras …", flush=True)
    thermal = read_csv(DATA_DIR / "thermal_cameras.csv")
    print(f"  {len(thermal)} cameras loaded.")

    # Attach colours and computed FoV
    assign_colors(rgb,     RGB_PALETTE)
    assign_colors(thermal, THERMAL_PALETTE)
    for cam in rgb + thermal:
        cam["fov_deg"] = compute_fov(cam)

    # Serialise for JavaScript
    camera_json    = build_camera_json(rgb, thermal)
    distances_json = json.dumps(DISTANCES_M)
    drones_json    = json.dumps(DRONE_DEFAULTS, indent=2)
    defaults_json  = json.dumps(THRESHOLD_DEFAULTS)

    # Embed Plotly.js
    print("Embedding Plotly.js (offline-capable) …", flush=True)
    plotlyjs = get_plotlyjs()
    print(f"  Plotly.js: {len(plotlyjs) / 1024:.0f} KB")

    # Build and write HTML
    print(f"Writing {OUTPUT_FILE} …", flush=True)
    html = build_html(plotlyjs, camera_json, distances_json, drones_json, defaults_json)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024:.0f} KB")
    print(f"\nDone.  Open in browser:\n  file://{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
