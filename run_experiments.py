from datetime import datetime
import os
import subprocess
from tqdm import tqdm

def run_experiment(session_file, transcript_tool, config, timestamp):
    """
    Runs the experiments.py script with the given session file and transcript tool.
    Config should be a tuple containing parameters to be passed to the experiments script.
    """
    config_params = map(str, config)  # Convert all config parameters to strings
    command = [
        'python',
        os.path.join(os.getcwd(), "experiments.py"),
        session_file,
        transcript_tool,
        *config_params,  # Unpack all configuration parameters into the command
        timestamp  # Ensure timestamp is passed as the last argument
    ]
    subprocess.run(command, shell=False, check=True)

def main():
    sessions_dir = os.path.join(os.getcwd(), "data", "sessions")
    transcript_tool = "whisperx"
    progress_file = "processed_sessions.txt"
    config_set = [
        (8000, 3800, 800, True, True, 3.0, 0.65, 0.6, 0.8, 0.8, 0.01, 0.05, 0.04),
        (8000, 3800, 800, True, True, 3.0, 0.65, 0.6, 0.8, 0.8, 0.01, 0.05, 0.05)
    ]

    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_sessions = {line.strip() for line in f}
    else:
        processed_sessions = set()

    session_files = [f for f in os.listdir(sessions_dir) if f.endswith(".wav")]

    for config in config_set:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for filename in tqdm(session_files, desc=f"Processing sessions for config {config}"):
            session_file = os.path.join(sessions_dir, filename)
            if session_file in processed_sessions:
                print(f"Skipping already processed session: {session_file}")
                continue

            run_experiment(session_file, transcript_tool, config, timestamp)
            with open(progress_file, 'a') as f:
                f.write(session_file + '\n')

if __name__ == "__main__":
    main()
