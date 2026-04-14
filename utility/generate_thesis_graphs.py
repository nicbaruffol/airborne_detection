import numpy as np
import matplotlib.pyplot as plt
import os

def generate_comparison_graph():
    # --- 1. Thesis-Specific Parameters ---
    drone_size_m = 0.300        # DJI Mini width
    rgb_width_px_1 = 1440       # Current RGB Sensor
    rgb_width_px_2 = 3840       # 4K UHD Sensor width
    
    # Define lenses by their horizontal Field of View (FOV)
    lenses_fov_deg = {
        'Wide Lens (55.3°)': 55.3,
        'Medium Lens (25.0°)': 25.0,
        'Telephoto Lens (10.0°)': 10.0,
        'Extreme Telephoto (5.0°)': 5.0
    }

    # Empirically Derived Thresholds
    detect_threshold = 35       # YOLO baseline limit
    track_threshold = 5         # Segmentation tracker limit

    # Separate Distances to evaluate
    distances_m_1 = np.linspace(1, 1000, 1000)  # 1 to 1000m for current sensor
    distances_m_2 = np.linspace(1, 3000, 3000)  # 1 to 3000m for 4K sensor

    # --- 2. Calculate Pixel Footprints ---
    def calculate_pixel_width(fov_deg, distance, sensor_width):
        fov_rad = np.radians(fov_deg)
        return (drone_size_m * sensor_width) / (2 * distance * np.tan(fov_rad / 2))

    # --- 3. Plotting Setup ---
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        try:
            plt.style.use('seaborn-paper')
        except OSError:
            plt.style.use('default')
            
    # Create 1 row, 2 columns. sharey=True keeps the y-axis locked between both plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    # --- Plot 1: Current 1440px Sensor (up to 1000m) ---
    ax1 = axes[0]
    for (label, fov), color in zip(lenses_fov_deg.items(), colors):
        pixels = calculate_pixel_width(fov, distances_m_1, rgb_width_px_1)
        ax1.plot(distances_m_1, pixels, label=label, color=color, linewidth=2)

    ax1.axhline(detect_threshold, color='black', linestyle='--', alpha=0.8, label=f'Detection Limit ({detect_threshold}px)')
    ax1.axhline(track_threshold, color='gray', linestyle='-.', alpha=0.8, label=f'Tracking Limit ({track_threshold}px)')
    
    ax1.set_title(f'Current RGB Sensor ({rgb_width_px_1}px)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Distance (meters)', fontsize=10)
    ax1.set_ylabel('Target Width in Image (pixels)', fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, 1000)
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=9)

    # --- Plot 2: 4K UHD Sensor (up to 3000m) ---
    ax2 = axes[1]
    for (label, fov), color in zip(lenses_fov_deg.items(), colors):
        pixels = calculate_pixel_width(fov, distances_m_2, rgb_width_px_2)
        ax2.plot(distances_m_2, pixels, label=label, color=color, linewidth=2)

    ax2.axhline(detect_threshold, color='black', linestyle='--', alpha=0.8, label=f'YOLO Fused Detection Limit ({detect_threshold}px)')
    ax2.axhline(track_threshold, color='gray', linestyle='-.', alpha=0.8, label=f'Tracking Limit ({track_threshold}px)')
    
    ax2.set_title(f'4K UHD Sensor ({rgb_width_px_2}px)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Distance (meters)', fontsize=10)
    # No ylabel needed here since it shares with ax1
    ax2.set_xlim(0, 3000)
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=9)

    # Add a main title for the whole figure
    fig.suptitle('Impact of Lens FOV and Sensor Resolution on Tracking Distances', fontsize=14, fontweight='bold', y=1.02)

    # --- 4. Export to PDF ---
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'lens_comparison_1440_vs_4k.pdf')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Graph successfully exported to: {output_path}")

if __name__ == "__main__":
    generate_comparison_graph()