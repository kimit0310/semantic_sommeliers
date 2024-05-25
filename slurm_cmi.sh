#!/bin/bash

#SBATCH --job-name=mobi_hbn_video_qa_run
#SBATCH --output=mobi_hbn_video_qa_run%j.out
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=6-23:59:00

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

# Split audio files into two groups
audio_folder="/data3/mobi/hbn_video_qa/extracted_audio"
audio_files=($(ls ${audio_folder}/*.wav))
total_files=${#audio_files[@]}
group_size=$((total_files / 2))
remainder=$((total_files % 2))

# First group
group1_files=("${audio_files[@]:0:${group_size}}")
group1_list="group1_list.txt"
for file in "${group1_files[@]}"; do
    echo "$(basename "$file")" >> ${group1_list}
done

# Second group
start_index=$group_size
end_index=$((start_index + group_size + remainder))
group2_files=("${audio_files[@]:${start_index}}")
group2_list="group2_list.txt"
for file in "${group2_files[@]}"; do
    echo "$(basename "$file")" >> ${group2_list}
done

# Run batch processes in parallel
srun --exclusive -N1 -n1 python batch_run.py --audio_list ${group1_list} &
srun --exclusive -N1 -n1 python batch_run.py --audio_list ${group2_list} &

wait

# Deactivate the environment at the end of the script
conda deactivate
