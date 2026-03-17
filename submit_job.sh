#!/bin/bash
#SBATCH --job-name=transform_train    # Name of your job
#SBATCH --time=24:00:00                  # Max runtime (Hours:Minutes:Seconds)
#SBATCH --ntasks=1                       # Number of tasks (your script is single-threaded)
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --mem-per-cpu=16G                 # Request 8GB of RAM
#SBATCH --gpus=rtx_4090:1  # Request 1 GPU
#SBATCH --output=transform_train_%j.log
#SBATCH --error=transform_train_%j.err   # Error log file
#SBATCH --tmp=700G   



# 1. Load the exact same modules you used to create your virtual environment
module load stack/.2024-05-silent  gcc/13.2.0 python/3.9.18
#module load       stack/2024-06  gcc/12.2.0 python_cuda/3.11.6
module load eth_proxy


# 2. Activate your virtual environment
source $HOME/airborne_detection/airbone/bin/activate

# 3. Navigate into the seg_tracker directory FIRST
cd /cluster/home/nbaruffol/airborne_detection/seg_tracker

# # 1. Define where the tar file is, and the target extraction folder
# NETWORK_TAR="/cluster/scratch/nbaruffol/airborne_dataset_new.tar"
# LOCAL_DIR="$TMPDIR/airborne_dataset_new"

# # 2. Extract the dataset directly to the node's local SSD
# echo "Starting data extraction to local SSD..."

# # -x extracts, -f specifies the file, and -C dictates where to unpack it
# tar -xf $NETWORK_TAR -C $TMPDIR

# echo "Data extraction complete!"

# # 3. Export the new fast path as an environment variable so Python can find it
# export FAST_DATA_DIR=$LOCAL_DIR

# 3. Run your Python script
python -u train_transformation.py check 030_tr_tsn_rn34_w3_crop_borders --epoch 500
echo "Check complete!"
python -u train_transformation.py export_model 030_tr_tsn_rn34_w3_crop_borders --epoch 500
echo "Model export complete!"
python -u train_transformation.py predict_dataset_offsets 030_tr_tsn_rn34_w3_crop_borders --part part1
echo "Prediction for part1 complete!"
python -u train_transformation.py predict_dataset_offsets 030_tr_tsn_rn34_w3_crop_borders --part part2
echo "Prediction for part2 complete!"
python -u train_transformation.py predict_dataset_offsets 030_tr_tsn_rn34_w3_crop_borders --part part3
echo "Prediction for part3 complete!"