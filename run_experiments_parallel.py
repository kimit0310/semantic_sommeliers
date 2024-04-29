import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

def run_experiment(session_file, transcript_tool, config, timestamp):
    """
    Runs the experiments.py script with the given session file and transcript tool.
    """
    config_params = map(str, config)  # Convert all config parameters to strings
    command = [
        'python',
        os.path.join(os.getcwd(), "experiments.py"),
        session_file,
        transcript_tool,
        *config_params,   # Unpack all configuration parameters into the command
        timestamp # MAKE SURE TIMESTAMP IS HERE
    ]
    subprocess.run(command, shell=False, check=True)

def main():
    sessions_dir = os.path.join(os.getcwd(), "data", "sessions")
    transcript_tool = "whisperx"  # Assuming you want to use 'whisperx' for all
    config_set = [
        (16000, 7500, 1000, True, True, 3.0, 0.65, 0.6, 0.8, 0.8, 0.01, 0.05, 0.05),
        (16000, 7500, 1000, True, True, 3.0, 0.65, 0.6, 0.8, 0.8, 0.01, 0.05, 0.04),
    ]

    # Create a unique timestamp for this batch run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Get a list of all .wav files in the sessions directory
    session_files = [f for f in os.listdir(sessions_dir) if f.endswith(".wav")]

    # Run experiments in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for config in config_set:
            for filename in session_files:
                session_file = os.path.join(sessions_dir, filename)
                futures.append(executor.submit(run_experiment, session_file, transcript_tool, config, timestamp))
        
        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()  # This will raise exceptions from the thread if any
            except Exception as exc:
                print(f"Generated an exception: {exc}")

if __name__ == "__main__":
    main()
