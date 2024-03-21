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

python experiments.py 5023141_speech_language_non.wav whisperx
python experiments.py 5030023_speech_language_non.wav whisperx
python experiments.py 5031633_speech_language.wav whisperx
python experiments.py 5156180_speech_language.wav whisperx
python experiments.py 5195387_speech_language_non.wav whisperx
python experiments.py 5253316_speech_language.wav whisperx
python experiments.py 5270706_speech_language.wav whisperx
python experiments.py 5282397_speech_language_non.wav whisperx
python experiments.py 5334942_speech_languageB.wav whisperx
python experiments.py 5413094_speech_language.wav whisperx
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
