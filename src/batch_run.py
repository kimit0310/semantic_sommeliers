"""
This module handles the batch processing of audio sessions using a specified transcription tool and configuration.
It includes functions for running experiments on individual sessions and a main function for managing the batch run.

Functions:
    - run_experiment: Run the experiment for a single audio session with the given configuration.
    - main: Main function to execute the batch processing pipeline. Parses arguments, reads the list of sessions,
            and processes each session file with the specified configurations.
"""

from datetime import datetime
import os
import subprocess
import argparse
from typing import Tuple
from tqdm import tqdm # type: ignore

def run_experiment(session_file: str, transcript_tool: str, config: Tuple, timestamp: str, error_log_path: str) -> None:
    """
    Run the experiment for a single audio session with the given configuration.

    Args:
        session_file (str): Path to the session file.
        transcript_tool (str): Name of the transcription tool.
        config (Tuple): Configuration parameters for the experiment.
        timestamp (str): Timestamp for the experiment run.
        error_log_path (str): Path to the error log file.

    Returns:
        None
    """
    config_params = map(str, config)
    command = [
        "python",
        os.path.join(os.getcwd(), "main.py"),
        session_file,
        transcript_tool,
        *config_params,
        timestamp,
    ]
    try:
        subprocess.run(command, shell=False, check=True)
    except subprocess.CalledProcessError as e:
        with open(error_log_path, "a") as error_log:
            error_log.write(f"Error processing {session_file}: {e}\n")

def main() -> None:
    """
    Main function to execute the batch processing pipeline. Parses arguments, reads the list of sessions,
    and processes each session file with the specified configurations.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Batch Run Script")
    parser.add_argument("--audio_list", type=str, required=True, help="Path to the list of audio files")
    parser.add_argument("--error_log", type=str, default="error_log.txt", help="Path to the error log file")
    args = parser.parse_args()

    with open(args.audio_list, "r") as f:
        session_files = [line.strip() for line in f.readlines()]

    transcript_tool = "whisperx"
    config_set = [
        (16000, 7500, 1000, True, True, 3.0, 0.65, 0.6, 0.8, 0.8, 0.01, 0.05, 0.05),
    ]

    for config in config_set:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for session_file in tqdm(session_files, desc="Processing Files"):
            run_experiment(session_file, transcript_tool, config, timestamp, args.error_log)

if __name__ == "__main__":
    main()