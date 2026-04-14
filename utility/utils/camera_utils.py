"""
Camera data helpers — CSV loading, FoV computation, colour assignment.

FoV formula (full horizontal FoV):
    FoV_h = 2 * arctan(sensor_width_mm / (2 * focal_length_mm))
    where sensor_width_mm = resolution_w * pixel_pitch_um / 1000

Note on nominal vs. effective focal length:
    Camera datasheets quote a *nominal* focal length (e.g. "9 mm").
    The actual *effective* focal length is typically 5–15 % shorter, which is
    why the computed FoV (e.g. 46.3° for Boson 640 + 9 mm lens) can differ
    from the spec-sheet HFOV (e.g. 50°).  Both the pixel-count and the FoV
    formulas use the same nominal focal length, so relative comparisons between
    cameras remain valid.
"""

import csv
import json
import math
import sys
from pathlib import Path


def read_csv(filepath: Path) -> list:
    """Parse a camera CSV file and return a list of typed row dicts."""
    if not filepath.exists():
        print(f"ERROR: CSV not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["focal_length_mm"] = float(row["focal_length_mm"])
            row["pixel_pitch_um"]  = float(row["pixel_pitch_um"])
            row["resolution_w"]    = int(row["resolution_w"])
            row["resolution_h"]    = int(row["resolution_h"])
            row["f_number"]        = float(row["f_number"])
            rows.append(row)
    return rows


def compute_fov(cam: dict) -> float:
    """
    Compute full horizontal FoV in degrees.

    FoV_h = 2 * arctan(sensor_width_mm / (2 * focal_length_mm))

    The factor-of-2 outside arctan converts the half-angle to the full angle.
    """
    sensor_w_mm = cam["resolution_w"] * cam["pixel_pitch_um"] / 1000.0
    fov_rad = 2.0 * math.atan(sensor_w_mm / (2.0 * cam["focal_length_mm"]))
    return round(math.degrees(fov_rad), 1)


def assign_colors(cameras: list, palette: list) -> list:
    """Attach a 'color' key to each camera dict, cycling through palette."""
    for i, cam in enumerate(cameras):
        cam["color"] = palette[i % len(palette)]
    return cameras


def build_camera_json(rgb: list, thermal: list) -> str:
    """Serialise both camera lists (with fov_deg attached) to a JSON string."""
    return json.dumps({"rgb": rgb, "thermal": thermal}, indent=2)


def read_radar_csv(filepath: Path) -> list:
    """Parse a radar sensor CSV file and return a list of typed row dicts."""
    if not filepath.exists():
        print(f"ERROR: CSV not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    float_cols = {"freq_ghz", "tx_power_dbm", "ant_gain_dbi",
                  "noise_figure_db", "bandwidth_mhz",
                  "beamwidth_h_deg", "beamwidth_v_deg"}
    rows = []
    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for col in float_cols:
                if col in row:
                    row[col] = float(row[col])
            rows.append(row)
    return rows
