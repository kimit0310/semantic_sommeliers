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
        "python",
        os.path.join(os.getcwd(), "main.py"),
        session_file,
        transcript_tool,
        *config_params,  # Unpack all configuration parameters into the command
        timestamp,  # Ensure timestamp is passed as the last argument
    ]
    subprocess.run(command, shell=False, check=True)


def main():
    sessions_dir = os.path.join(os.getcwd(), "data", "sessions")
    transcript_tool = "whisperx"
    config_set = [
        (16000, 7500, 1000, True, True, 3.0, 0.65, 0.6, 0.8, 0.8, 0.01, 0.05, 0.05),
        (16000, 7500, 1000, True, True, 3.0, 0.65, 0.6, 0.8, 0.8, 0.01, 0.05, 0.04),
    ]

    session_files = [f for f in os.listdir(sessions_dir) if f.endswith(".wav")]

    for config in config_set:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for filename in tqdm(session_files):
            session_file = os.path.join(sessions_dir, filename)

            run_experiment(session_file, transcript_tool, config, timestamp)


if __name__ == "__main__":
    main()
