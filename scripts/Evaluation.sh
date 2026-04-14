#!/bin/bash
#SBATCH --job-name=Evaluation
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=Evaluation_%j.log
#SBATCH --error=Evaluation_%j.err  

set -e

module load stack/.2024-05-silent  gcc/13.2.0 python/3.9.18
module load eth_proxy

source $HOME/airborne_detection/airbone/bin/activate

# 1. Navigate to the codebase
# cd /cluster/home/nbaruffol/airborne_detection/seg_tracker

# # 1. Setup paths
# TAR_FILE="/cluster/scratch/nbaruffol/airborne_dataset_new_with_transforms.tar"
# LOCAL_DIR="$TMPDIR"

# # 2. Copy the .tar archive to the fast SSD
# echo "Copying .tar to fast SSD..."
# rsync -aq $TAR_FILE $LOCAL_DIR/

# # 3. Extract the archive on the SSD! (THIS IS THE MAGIC STEP)
# echo "Extracting dataset..."
# cd $LOCAL_DIR
# tar -xf airborne_dataset_new_with_transforms.tar

# # 4. Point Python to the newly extracted folder
# export FAST_DATA_DIR="$LOCAL_DIR/airborne_dataset_new"

# echo "Clearing old cache files..."
# rm -f $FAST_DATA_DIR/*.pkl 
cd /cluster/home/nbaruffol/airborne_detection/scripts/
cd /cluster/home/nbaruffol/airborne_detection/seg_tracker

# # RGB-only YOLO
# python inference.py --mode yolo_rgb --yolo_weights /cluster/home/nbaruffol/airborne_detection/output/models/rgb_best.pt --video /cluster/scratch/nbaruffol/raw_videos/rgb_fixed_1.mp4

# # IR-only YOLO
# python inference.py --mode yolo_ir --yolo_weights /cluster/home/nbaruffol/airborne_detection/output/models/ir_best.pt --ir_video /cluster/scratch/nbaruffol/raw_videos/ir_fixed_1.mp4

# Fused RGB-T YOLO
# python inference.py --mode yolo_rgbt --yolo_weights /cluster/home/nbaruffol/YOLOv11-RGBT/runs/Anti-UAV/yolo11n-RGBRGB2/weights/last.pt --video /cluster/scratch/nbaruffol/raw_videos/rgb_fixed_1.mp4 --ir_video /cluster/scratch/nbaruffol/raw_videos/ir_fixed_1.mp4

# python inference.py --mode yolo_csv_rgb --det_csv /cluster/home/nbaruffol/airborne_detection/seg_tracker/final_yolo/rgb_only_detections.csv
# python inference.py --mode yolo_csv_ir --det_csv /cluster/home/nbaruffol/airborne_detection/seg_tracker/final_yolo/ir_only_detections.csv
# python inference.py --mode yolo_csv_rgbt --det_csv /cluster/home/nbaruffol/airborne_detection/seg_tracker/final_yolo/fused_detections.csv
# python inference.py --mode yolo_csv_compare --rgb_csv /cluster/home/nbaruffol/airborne_detection/seg_tracker/final_yolo/rgb_only_detections.csv --ir_csv /cluster/home/nbaruffol/airborne_detection/seg_tracker/final_yolo/ir_only_detections.csv --fused_csv /cluster/home/nbaruffol/airborne_detection/seg_tracker/final_yolo/fused_detections.csv

python inference.py --mode evaluate_rgb_baseline --video /cluster/scratch/nbaruffol/raw_videos/rgb_fixed_1.mp4 --exp 120_hrnet32_all
python inference.py --mode evaluate_fused \
    --video /cluster/scratch/nbaruffol/raw_videos/rgb_fixed_1.mp4 \
    --ir_video /cluster/scratch/nbaruffol/raw_videos/ir_fixed_1.mp4 \
    --fused_weights /cluster/home/nbaruffol/airborne_detection/output/models/120_hrnet32_fused_2220.pth \
    --conf_threshold 0.15
python inference.py --mode evaluate_ir --ir_video /cluster/scratch/nbaruffol/raw_videos/ir_fixed_1.mp4 --exp 120_hrnet32_all
python inference.py --mode evaluate_rgb --exp 120_hrnet32_all
python inference.py --mode visualize --video /cluster/scratch/nbaruffol/raw_videos/rgb_fixed_3.mp4 --exp 120_hrnet32_all



echo "All done!"