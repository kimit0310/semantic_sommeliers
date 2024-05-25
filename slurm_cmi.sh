#!/bin/bash

#SBATCH --job-name=mobi_hbn_video_qa_run
#SBATCH --output=mobi_hbn_video_qa_run%j.out
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --nodes=1

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

# Split audio files into three groups
audio_folder="/data3/mobi/hbn_video_qa/extracted_audio"
audio_files=($(ls ${audio_folder}/*.wav))
total_files=${#audio_files[@]}
group_size=$((total_files / 3))
remainder=$((total_files % 3))

# First group
group1_files=("${audio_files[@]:0:${group_size}}")
group1_list="group1_list.txt"
printf "%s\n" "${group1_files[@]}" > ${group1_list}

# Second group
group2_files=("${audio_files[@]:${group_size}:${group_size}}")
group2_list="group2_list.txt"
printf "%s\n" "${group2_files[@]}" > ${group2_list}

# Third group
group3_files=("${audio_files[@]:$((2 * group_size)):${group_size + remainder}}")
group3_list="group3_list.txt"
printf "%s\n" "${group3_files[@]}" > ${group3_list}

# Run batch processes in parallel
srun -N1 -n1 python batch_run.py --audio_list ${group1_list} &
srun -N1 -n1 python batch_run.py --audio_list ${group2_list} &
srun -N1 -n1 python batch_run.py --audio_list ${group3_list} &

wait

conda deactivate
