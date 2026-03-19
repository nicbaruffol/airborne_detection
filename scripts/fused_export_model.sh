#!/bin/bash
#SBATCH --job-name=fused_export_model    # Name of your job
#SBATCH --time=2:00:00           # Max runtime (Hours:Minutes:Seconds)
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8        # Number of CPU cores
#SBATCH --mem-per-cpu=8G         # Request 12GB of RAM per CPU
#SBATCH --gpus=rtx_4090:1         # Request 1 GPU
#SBATCH --output=fused_export_model_%j.log
#SBATCH --error=fused_export_model_%j.err 
#SBATCH --tmp=50G                

# 1. Load the exact same modules you used to create your virtual environment
module load stack/.2024-05-silent gcc/13.2.0 python/3.9.18
module load eth_proxy

# 2. Activate your virtual environment
source $HOME/airborne_detection/airbone/bin/activate

# 3. Navigate to where train.py is located
# (Double check this is the right folder! Sometimes it's in /utility or the root folder)
cd /cluster/home/nbaruffol/airborne_detection/seg_tracker

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


# 7. Run your Python script
echo "starting check..."
python train.py check 120_hrnet32_fused --fold 0 --epoch 2220

echo "Starting export..."
python train.py export_model 120_hrnet32_fused --epoch 2220

echo "starting prediction..."
python predict_oof.py predict 120_hrnet32_fused --fold 0 --epoch 2220 \
  --part part1 --flights 5 --from_flight 0 --step 1

echo "rendering videos..."
python visualize_predictions.py 120_hrnet32_fused --epoch 2220 --conf 0.5 --flights 5 --part part1

