#!/bin/bash

#SBATCH --job-name=mobi_hbn_video_qa_run
#SBATCH --output=mobi_hbn_video_qa_run%j.out
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00

# Define variables for paths
CONDA_DIR="/home/$USER/miniconda"
CONDA_SH="$CONDA_DIR/etc/profile.d/conda.sh"
CONDA_ENV_NAME="hbn_qa"
FFMPEG_DIR="/home/$USER/ffmpeg"

# Set up Conda path and initialize
export PATH="$CONDA_DIR/bin:$PATH"
source $CONDA_SH

# Activate the environment
source activate $CONDA_ENV_NAME

# Add ffmpeg to PATH
export PATH="$FFMPEG_DIR:$PATH"

# Navigate to the project directory
cd /home/$USER/semantic_sommeliers

# Run batch processes in parallel
srun --exclusive -N1 -n1 --gpus-per-task=1 --cpus-per-task=16 --mem=40G python batch_run.py --audio_list new_group1_list.txt --error_log error_log_part1.txt &
srun --exclusive -N1 -n1 --gpus-per-task=1 --cpus-per-task=16 --mem=40G python batch_run.py --audio_list new_group2_list.txt --error_log error_log_part2.txt &

wait

# Deactivate the environment at the end of the script
conda deactivate
