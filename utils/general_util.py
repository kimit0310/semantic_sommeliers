import argparse
import csv
import glob
import json
import os
from typing import Dict, Any, List, Tuple

from matplotlib import pyplot as plt
import numpy as np
from src.config import Config

def generate_json(input_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a JSON object with concatenated text and timestamps from input JSON.

    Args:
        input_json (dict): The input JSON object containing segments and words.

    Returns:
        dict: A JSON object containing concatenated text and word chunks with timestamps.
    """
    segments = input_json["segments"]
    concatenated_text = ""
    chunks = []

    for segment in segments:
        for word_info in segment["words"]:
            concatenated_text += word_info["word"] + " "
            start = word_info.get("start")
            end = word_info.get("end")
            if start is not None and end is not None:
                chunks.append({"text": word_info["word"], "timestamp": [start, end]})
            else:
                print(f"Warning: Missing timestamp for word '{word_info['word']}'")

    concatenated_text = concatenated_text.strip()

    output_json = {"text": concatenated_text, "chunks": chunks}

    return output_json

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments including hyperparameters with defaults from Config.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Experiment script for session processing."
    )
    parser.add_argument("session_name", help="Name of the session")
    parser.add_argument("transcript_tool", help="Name of the transcript tool")
    parser.add_argument(
        "new_sample_rate",
        type=int,
        default=Config.new_sample_rate,
        help="New sample rate for audio processing",
        nargs="?",
    )
    parser.add_argument(
        "highcut",
        type=int,
        default=Config.highcut,
        help="Highcut frequency for filtering",
        nargs="?",
    )
    parser.add_argument(
        "lowcut",
        type=int,
        default=Config.lowcut,
        help="Lowcut frequency for filtering",
        nargs="?",
    )
    parser.add_argument(
        "normalization",
        type=bool,
        default=Config.normalization,
        help="Enable or disable normalization",
        nargs="?",
    )
    parser.add_argument(
        "filtering",
        type=bool,
        default=Config.filtering,
        help="Enable or disable filtering",
        nargs="?",
    )
    parser.add_argument(
        "seconds_threshold",
        type=float,
        default=Config.seconds_threshold,
        help="Threshold in seconds for audio processing",
        nargs="?",
    )
    parser.add_argument(
        "story_absolute_peak_height",
        type=float,
        default=Config.story_absolute_peak_height,
        help="Absolute peak height for story detection",
        nargs="?",
    )
    parser.add_argument(
        "long_instructions_peak_height",
        type=float,
        default=Config.long_instructions_peak_height,
        help="Peak height for long instructions",
        nargs="?",
    )
    parser.add_argument(
        "word_instructions_peak_height",
        type=float,
        default=Config.word_instructions_peak_height,
        help="Peak height for word instructions",
        nargs="?",
    )
    parser.add_argument(
        "non_word_instructions_peak_height",
        type=float,
        default=Config.non_word_instructions_peak_height,
        help="Peak height for non-word instructions",
        nargs="?",
    )
    parser.add_argument(
        "long_instructions_absolute_peak_height",
        type=float,
        default=Config.long_instructions_absolute_peak_height,
        help="Absolute peak height for long instructions",
        nargs="?",
    )
    parser.add_argument(
        "word_instructions_absolute_peak_height",
        type=float,
        default=Config.word_instructions_absolute_peak_height,
        help="Absolute peak height for word instructions",
        nargs="?",
    )
    parser.add_argument(
        "non_word_instructions_absolute_peak_height",
        type=float,
        default=Config.non_word_instructions_absolute_peak_height,
        help="Absolute peak height for non-word instructions",
        nargs="?",
    )
    parser.add_argument(
        "timestamp", help="Timestamp for the batch run", default="single", nargs="?"
    )
    return parser.parse_args()

def setup_directories(base_dir: str, config: Config, timestamp: str) -> Tuple[str, str, str, str]:
    """
    Set up directories for the current run, creating unique folders based on the timestamp.

    Args:
        base_dir (str): The base directory for the run.
        config (Config): Configuration object.
        timestamp (str): Timestamp for the run.

    Returns:
        tuple: Paths to labels, transcriptions, similarity, and cross_correlation folders.
    """
    data_folder = os.path.join(base_dir, f"data_run_{timestamp}")

    labels_folder = os.path.join(data_folder, "labels")
    transcriptions_folder = os.path.join(data_folder, "transcriptions")
    similarity_folder = os.path.join(data_folder, "similarity")
    cross_correlations_folder = os.path.join(data_folder, "cross_correlation")

    os.makedirs(labels_folder, exist_ok=True)
    os.makedirs(transcriptions_folder, exist_ok=True)
    os.makedirs(similarity_folder, exist_ok=True)
    os.makedirs(cross_correlations_folder, exist_ok=True)

    config_path = os.path.join(data_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return (
        labels_folder,
        transcriptions_folder,
        similarity_folder,
        cross_correlations_folder,
    )

def save_labels_file(labels_folder: str, session_file_path: str, instructions_timings: List[Dict[str, Any]]) -> None:
    """
    Save the instructions timings to a labels(.txt) file.

    Args:
        labels_folder (str): Path to the labels folder.
        session_file_path (str): Path to the session file.
        instructions_timings (list): List of instruction timings.

    Returns:
        None
    """
    os.makedirs(labels_folder, exist_ok=True)
    with open(
        os.path.join(labels_folder, os.path.basename(session_file_path)[:-4] + ".txt"),
        "w",
        encoding="utf-8",
    ) as file:
        for instructions_timing in instructions_timings:
            if instructions_timing is not None:
                file.write(
                    f"{instructions_timing['start']}\t{instructions_timing['start'] + instructions_timing['duration']}\t{instructions_timing['label']}\n"
                )
    return

def get_instructions(instructions_folder: str) -> List[str]:
    """
    Get a sorted list of instruction file paths.

    Args:
        instructions_folder (str): Path to the instructions folder.

    Returns:
        list: Sorted list of instruction file paths.
    """
    instruction_file_paths = sorted(
        [
            os.path.join(instructions_folder, instruction_filename)
            for instruction_filename in os.listdir(instructions_folder)
            if instruction_filename.endswith(".wav")
        ]
    )
    first_10_instructions = instruction_file_paths[:10]
    twenty_first_instruction = instruction_file_paths[20:21]
    last_instruction = instruction_file_paths[-1:]
    remaining_instructions = (
        instruction_file_paths[10:20] + instruction_file_paths[21:38]
    )
    instruction_file_paths = (
        first_10_instructions
        + twenty_first_instruction
        + last_instruction
        + remaining_instructions
    )
    return instruction_file_paths

def get_sessions(sessions_folder: str) -> List[str]:
    """
    Get a sorted list of session file paths.

    Args:
        sessions_folder (str): Path to the sessions folder.

    Returns:
        list: Sorted list of session file paths.
    """
    session_file_paths = sorted(
        [
            os.path.join(sessions_folder, session_filename)
            for session_filename in os.listdir(sessions_folder)
            if session_filename.endswith(".wav")
        ]
    )
    return session_file_paths


def min_max_normalization(arr: Any) -> Any:
    """
    Normalize an array using min-max normalization.

    Args:
        arr (Any): Array to normalize.

    Returns:
        Any: Normalized array.
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr


def save_cross_correlation(cross_correlation: Any, peaks_indices: List[int], time_values: Any, file_path: str) -> None:
    """
    Save the cross-correlation plot to a file.

    Args:
        cross_correlation (Any): Cross-correlation data.
        peaks_indices (list): Indices of the peaks in the cross-correlation.
        time_values (Any): Time values for the cross-correlation.
        file_path (str): Path to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, cross_correlation, label="Cross-Correlation")
    plt.plot(
        time_values[peaks_indices],
        cross_correlation[peaks_indices],
        "o",
        label="Top Peaks",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Cross-Correlation")
    plt.title("Cross-Correlation between Session and Instruction")
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

def plot_cross_correlation(
    session_file_path: str,
    instruction_file_path: str,
    standardized_normalized_cross_corr: Any,
    peaks_indices: List[int],
    cross_correlations_folder: str,
    sr: int,
    my_new_session_start: int,
) -> None:
    """
    Generate and save the cross-correlation plot for a session.

    Args:
        session_file_path (str): Path to the session file.
        instruction_file_path (str): Path to the instruction file.
        standardized_normalized_cross_corr (Any): Normalized cross-correlation data.
        peaks_indices (list): Indices of the peaks in the cross-correlation.
        cross_correlations_folder (str): Path to the cross-correlations folder.
        sr (int): Sample rate.
        my_new_session_start (int): Start time for the session.

    Returns:
        None
    """
    cross_correlations_session_folder = os.path.join(
        cross_correlations_folder, os.path.basename(session_file_path)[:-4]
    )
    os.makedirs(cross_correlations_session_folder, exist_ok=True)
    cross_correlation_figure_file_name = f"{os.path.basename(instruction_file_path)[:-4]}_{os.path.basename(session_file_path)[:-4]}.png"

    time_values = (
        np.arange(len(standardized_normalized_cross_corr)) / sr
        + my_new_session_start / sr
    )

    save_cross_correlation(
        standardized_normalized_cross_corr,
        peaks_indices,
        time_values,
        os.path.join(
            cross_correlations_session_folder, cross_correlation_figure_file_name
        ),
    )

def finalize_results(
    labels_folder: str, session_file_path: str, story_timings: List[Dict[str, Any]], instructions_timings: List[Dict[str, Any]]
) -> None:
    """
    Save the final timings to a labels file.

    Args:
        labels_folder (str): Path to the labels folder.
        session_file_path (str): Path to the session file.
        story_timings (list): List of story timings.
        instructions_timings (list): List of instruction timings.

    Returns:
        None
    """
    all_timings = instructions_timings + story_timings
    save_labels_file(labels_folder, session_file_path, all_timings)


def summarize_results(labels_folder: str) -> None:
    """
    Summarize the results of the session processing and save to a CSV file.

    Args:
        labels_folder (str): Path to the labels folder.

    Returns:
        None
    """
    session_summaries = []

    label_files = glob.glob(os.path.join(labels_folder, "*.txt"))

    total_audio_instructions = [f"{i:02d}" for i in range(38)]
    total_stories = ["story_0", "story_1"]

    for file_path in label_files:
        session_name = os.path.basename(file_path).replace(".txt", "")
        detected_audio = set()
        detected_stories = set()

        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                label = parts[2]

                if label in total_stories:
                    detected_stories.add(label)
                elif "_" in label:
                    audio_index = label.split("_")[0]
                    detected_audio.add(audio_index)

        missed_audio = sorted(
            set(total_audio_instructions) - detected_audio, key=lambda x: int(x)
        )
        missed_stories = sorted(set(total_stories) - detected_stories)

        session_summaries.append(
            {
                "Session": session_name,
                "Total Audio Instructions Detected": len(detected_audio),
                "Total Stories Detected": len(detected_stories),
                "Missed Audio Instructions": ", ".join(missed_audio),
                "Missed Stories": ", ".join(missed_stories),
            }
        )

    csv_file_path = os.path.join(labels_folder, "summary_report.csv")
    with open(csv_file_path, "w", newline='') as csvfile:
        fieldnames = [
            "Session",
            "Total Audio Instructions Detected",
            "Total Stories Detected",
            "Missed Audio Instructions",
            "Missed Stories",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for summary in session_summaries:
            writer.writerow(summary)
