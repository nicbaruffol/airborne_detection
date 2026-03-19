#!/bin/bash
#SBATCH --job-name=fused_video   # Name of your job
#SBATCH --time=2:00:00           # Max runtime (Hours:Minutes:Seconds)
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8        # Number of CPU cores
#SBATCH --mem-per-cpu=8G         # Request 12GB of RAM per CPU
#SBATCH --gpus=rtx_4090:1         # Request 1 GPU
#SBATCH --output=fused_video_%j.log
#SBATCH --error=fused_video_%j.err 
#SBATCH --tmp=50G                

# 1. Load the exact same modules you used to create your virtual environment
module load stack/.2024-05-silent gcc/13.2.0 python/3.9.18
module load eth_proxy

# 2. Activate your virtual environment
source $HOME/airborne_detection/airbone/bin/activate

# 3. Navigate to where train.py is located
# (Double check this is the right folder! Sometimes it's in /utility or the root folder)
cd /cluster/home/nbaruffol/airborne_detection/

# # 4. Define where the tar file is, and the target extraction folder
# NETWORK_TAR="/cluster/scratch/nbaruffol/anti_uav_formatted.tar"
# LOCAL_DIR="$TMPDIR/anti_uav_formatted"

# # 5. Extract the dataset directly to the node's local SSD
# echo "Starting data extraction to local SSD..."
# tar -xf $NETWORK_TAR -C $TMPDIR
# echo "Data extraction complete! Verifying contents:"
# ls -lh $LOCAL_DIR/part1/ImageSets

# # 6. Export the new fast path as an environment variable so Python can find it
export FAST_DATA_DIR="/cluster/scratch/nbaruffol/anti_uav_formatted"

python run_my_video.py \
  --rgb  /cluster/scratch/nbaruffol/raw_videos/rgb_fixed_1.mp4 \
  --ir   /cluster/scratch/nbaruffol/raw_videos/ir_fixed_1.mp4 \
  --output /cluster/scratch/nbaruffol/raw_videos/fused_tracked_1.mp4 \
  --conf 0.5