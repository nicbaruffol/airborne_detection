#!/bin/bash
#SBATCH --job-name=predict_offsets
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=predict_offsets_%j.log
#SBATCH --error=predict_offsets_%j.err
#SBATCH --tmp=700G   

set -e

module load stack/.2024-05-silent  gcc/13.2.0 python/3.9.18
module load eth_proxy

source $HOME/airborne_detection/airbone/bin/activate

# 1. Navigate to the codebase
cd /cluster/home/nbaruffol/airborne_detection/seg_tracker

# 2. Setup paths
NETWORK_DIR="/cluster/scratch/nbaruffol/airborne_dataset_new"
LOCAL_DIR="$TMPDIR/airborne_dataset_new"

# 3. Copy dataset to the fast SSD so the GPU doesn't starve
echo "Copying dataset to fast SSD (This takes a few minutes)..."
mkdir -p $LOCAL_DIR
rsync -aq $NETWORK_DIR/ $LOCAL_DIR/
export FAST_DATA_DIR=$LOCAL_DIR

echo "====================================================="
echo "VERIFYING CONFIG FOR PART 1..."
python -c "import config; print(f'---> ALERT: USING MODEL EPOCH {config.TRANSFORM_MODEL_EPOCH} <---')"
echo "Starting predictions for Part 1..."
python -u train_transformation.py predict_dataset_offsets 030_tr_tsn_rn34_w3_crop_borders --part part1

echo "====================================================="
echo "VERIFYING CONFIG FOR PART 2..."
python -c "import config; print(f'---> ALERT: USING MODEL EPOCH {config.TRANSFORM_MODEL_EPOCH} <---')"
echo "Starting predictions for Part 2..."
python -u train_transformation.py predict_dataset_offsets 030_tr_tsn_rn34_w3_crop_borders --part part2

echo "====================================================="
echo "VERIFYING CONFIG FOR PART 3..."
python -c "import config; print(f'---> ALERT: USING MODEL EPOCH {config.TRANSFORM_MODEL_EPOCH} <---')"
echo "Starting predictions for Part 3..."
python -u train_transformation.py predict_dataset_offsets 030_tr_tsn_rn34_w3_crop_borders --part part3

# 5. THE MOST IMPORTANT STEP: Rescue the results from the temporary drive!
echo "Predictions complete! Copying the mathematical offsets back to permanent storage..."
rsync -av $LOCAL_DIR/frame_transforms/ $NETWORK_DIR/frame_transforms/

echo "All done!"