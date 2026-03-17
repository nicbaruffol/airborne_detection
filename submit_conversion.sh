#!/bin/bash
#SBATCH --job-name=prepare_anti_uav    # Name of your job
#SBATCH --time=4:00:00                  # Max runtime (Hours:Minutes:Seconds)
#SBATCH --ntasks=1                       # Number of tasks (your script is single-threaded)
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --mem-per-cpu=8G                 # Request 8GB of RAM
#SBATCH --output=prepare_anti_  uav_%j.log
#SBATCH --error=prepare_anti_uav_%j.err   # Error log file


# 1. Load the exact same modules you used to create your virtual environment
module load stack/.2024-05-silent  gcc/13.2.0 python/3.9.18
#module load       stack/2024-06  gcc/12.2.0 python_cuda/3.11.6
module load eth_proxy


# 2. Activate your virtual environment
source $HOME/airborne_detection/airbone/bin/activate

# 3. Navigate into the seg_tracker directory FIRST
cd /cluster/home/nbaruffol/airborne_detection/seg_tracker

# 3. Run your Python script
FAST_DATA_DIR=/cluster/scratch/nbaruffol/anti_uav_formatted 
python train_transformation.py predict_dataset_offsets 030_tr_tsn_rn34_w3_crop_borders --part part1