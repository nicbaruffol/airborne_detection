#!/bin/bash
#SBATCH --job-name=RGB-IR_tracking
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=RGB-IR_tracking_%j.log
#SBATCH --error=RGB-IR_tracking_%j.err  

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

cd /cluster/home/nbaruffol/airborne_detection/seg_tracker
python inference.py --mode evaluate_ir_classical --ir_video /cluster/scratch/nbaruffol/raw_videos/ir_fixed_1.mp4
# python inference.py --mode evaluate_ir --ir_video /cluster/scratch/nbaruffol/raw_videos/ir_fixed_1.mp4 --exp 120_hrnet32_all
# python inference.py --mode evaluate_rgb --exp 120_hrnet32_all
# python inference.py --mode visualize --video /cluster/scratch/nbaruffol/raw_videos/rgb_fixed_1.mp4 --exp 120_hrnet32_all


echo "All done!"