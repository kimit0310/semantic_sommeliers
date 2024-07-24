"""
#!/bin/bash       
#SBATCH --job-name=cmi
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fabiocat@mit.edu
#SBATCH -t 36:00:00          # walltime = 1 hours and 30 minutes
#SBATCH --mem=20G 
#SBATCH -N 1
#SBATCH -n 6 
#SBATCH --gres=gpu:1
#SBATCH -p gablab

module load openmind/ffmpeg/20160310
eval "$(conda shell.bash hook)"
conda activate cmi

python experiments.py session1.wav whisperx
python experiments.py session2.wav whisperx
python experiments.py session3.wav whisperx
"""

#!/bin/bash       
#SBATCH --job-name=cmi
#SBATCH --output=./logs/%A_%a.out
#SBATCH --error=./logs/%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fabiocat@mit.edu
#SBATCH -t 5:00:00
#SBATCH --mem=20G 
#SBATCH -N 1
#SBATCH -n 6 
#SBATCH --gres=gpu:1
#SBATCH -p gablab
#SBATCH --array=1-10

module load openmind/ffmpeg/20160310
ffmpeg

eval "$(conda shell.bash hook)"
conda activate cmi

# Function to execute python script for a specific file
run_python_script() {
    wav_file=$(ls data/sessions/*.wav | sed -n ${SLURM_ARRAY_TASK_ID}p)
    wav_basename=$(basename "$wav_file")
    python experiments.py "$wav_basename" whisperx
}

# Execute python script for the corresponding wav file in the array
run_python_script
