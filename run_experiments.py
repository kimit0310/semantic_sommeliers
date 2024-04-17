import os
import subprocess
from tqdm import tqdm

def run_experiment(session_file, transcript_tool, config):
    """
    Runs the experiments.py script with the given session file and transcript tool.
    Config should be a tuple containing (new_sample_rate, highcut, lowcut).
    """
    new_sample_rate, highcut, lowcut = map(str, config)  # Convert all to string
    command = [
        'python',
        os.path.join(os.getcwd(), "experiments.py"),
        session_file,
        transcript_tool,
        new_sample_rate,
        highcut,
        lowcut
    ]
    subprocess.run(command, shell=False, check=True)

def main():
    sessions_dir = os.path.join(os.getcwd(), "data", "sessions")
    transcript_tool = "whisperx"  # Assuming you want to use 'whisperx' for all
    progress_file = "processed_sessions.txt"  # File to track processed sessions
    config_set = [(8000, 3000, 512), (16000, 4000, 256)]  # Example configurations

    # Load the set of already processed sessions
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_sessions = {line.strip() for line in f}
    else:
        processed_sessions = set()

    # Get a list of all .wav files in the sessions directory
    session_files = [f for f in os.listdir(sessions_dir) if f.endswith(".wav")]

    # Iterate through all configurations
    for config in config_set:
        for filename in tqdm(session_files, desc=f"Processing sessions for config {config}"):
            session_file = os.path.join(sessions_dir, filename)

            # Check if this session has already been processed
            if session_file in processed_sessions:
                print(f"Skipping already processed session: {session_file}")
                continue

            run_experiment(session_file, transcript_tool, config)

            # Save this session as processed
            with open(progress_file, 'a') as f:
                f.write(session_file + '\n')

if __name__ == "__main__":
    main()
