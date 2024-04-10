import os
import subprocess
from tqdm import tqdm

def run_experiment(session_file, transcript_tool):
    """
    Runs the experiments.py script with the given session file and transcript tool.
    """
    command = [
        'python',
        os.path.join(os.getcwd(), "experiments.py"),
        session_file,
        transcript_tool
    ]
    subprocess.run(command, shell=False, check=True)
def main():
    sessions_dir = os.path.join(os.getcwd(), "data", "sessions")
    transcript_tool = "whisperx"  # Assuming you want to use 'whisperx' for all
    progress_file = "processed_sessions.txt"  # File to track processed sessions

    # Load the set of already processed sessions
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_sessions = {line.strip() for line in f}
    else:
        processed_sessions = set()

    # Get a list of all .wav files in the sessions directory
    session_files = [f for f in os.listdir(sessions_dir) if f.endswith(".wav")]

    # Iterate through all files in the sessions directory with tqdm for progress
    for filename in tqdm(session_files, desc="Processing sessions"):
        session_file = os.path.join(sessions_dir, filename)

        # Check if this session has already been processed
        if session_file in processed_sessions:
            print(f"Skipping already processed session: {session_file}")
            continue

        print(f"\nRunning experiment for session: {session_file}")
        run_experiment(session_file, transcript_tool)

        # Save this session as processed
        with open(progress_file, 'a') as f:
            f.write(session_file + '\n')

if __name__ == "__main__":
    main()
