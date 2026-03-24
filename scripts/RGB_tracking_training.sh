#!/bin/bash
#SBATCH --job-name=RGB_tracking
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=RGB_tracking_%j.log
#SBATCH --error=RGB_tracking_%j.err
#SBATCH --tmp=700G   

set -e

module load stack/.2024-05-silent  gcc/13.2.0 python/3.9.18
module load eth_proxy

source $HOME/airborne_detection/airbone/bin/activate

# 1. Navigate to the codebase
# cd /cluster/home/nbaruffol/airborne_detection/seg_tracker

# 1. Setup paths
TAR_FILE="/cluster/scratch/nbaruffol/airborne_dataset_new_with_transforms.tar"
LOCAL_DIR="$TMPDIR"

# 2. Copy the .tar archive to the fast SSD
echo "Copying .tar to fast SSD..."
rsync -aq $TAR_FILE $LOCAL_DIR/

# 3. Extract the archive on the SSD! (THIS IS THE MAGIC STEP)
echo "Extracting dataset..."
cd $LOCAL_DIR
tar -xf airborne_dataset_new_with_transforms.tar

# 4. Point Python to the newly extracted folder
export FAST_DATA_DIR="$LOCAL_DIR/airborne_dataset_new"

echo "Clearing old cache files..."
rm -f $FAST_DATA_DIR/*.pkl 

cd /cluster/home/nbaruffol/airborne_detection/seg_tracker
python -u train.py train 120_hrnet32_all

echo "All done!"