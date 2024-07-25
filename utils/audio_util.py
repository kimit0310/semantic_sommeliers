import math
import os
import numpy as np
import torch
import torchaudio # type: ignore
import torchaudio.transforms as T # type: ignore
import pyloudnorm as pyln # type: ignore
from typing import Tuple, List, Optional


from src.config import Config
from scipy.signal import butter, correlate, find_peaks, lfilter # type: ignore

from utils.general_util import min_max_normalization, plot_cross_correlation


def load_audio(
    path: str,
    new_sample_rate: Optional[int] = None,
    filtering: bool = False,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    normalization: bool = False,
    max_length: Optional[float] = None,
) -> Tuple[torch.Tensor, int, float]:
    """
    Load and preprocess audio from a given file path.

    Args:
        path (str): Path to the audio file.
        new_sample_rate (Optional[int], optional): New sample rate for resampling.
        filtering (bool, optional): Whether to apply bandpass filtering.
        lowcut (Optional[float], optional): Low cut-off frequency for filtering.
        highcut (Optional[float], optional): High cut-off frequency for filtering.
        normalization (bool, optional): Whether to normalize the volume.
        max_length (Optional[float], optional): Maximum length of the audio in seconds.

    Returns:
        Tuple[torch.Tensor, int, float]: Processed waveform, sample rate, and duration of the instruction.
    """
    waveform, sample_rate = torchaudio.load(path)  # type: ignore
    instruction_duration = waveform.shape[1] / sample_rate

    if new_sample_rate:
        resampler = T.Resample(sample_rate, new_sample_rate, dtype=waveform.dtype)
        waveform = resampler(waveform)
        sample_rate = new_sample_rate
    if filtering:
        if lowcut is not None and highcut is not None:
            waveform = torch.tensor(
                bandpass_filter(waveform, lowcut=lowcut, highcut=highcut, fs=sample_rate)
            )
    if normalization:
        waveform = torch.tensor(
            normalize_volume(waveform[0].numpy(), sample_rate).reshape(1, -1)
        )
    waveform = waveform.squeeze()
    if max_length is not None:
        waveform = waveform[: int(max_length * sample_rate)]
    return waveform, sample_rate, instruction_duration

def normalize_volume(data: np.ndarray, rate: int, new_loudness: float = -23) -> np.ndarray:
    """
    Normalize the volume of an audio signal.

    Args:
        data (np.ndarray): Audio data.
        rate (int): Sample rate.
        new_loudness (float, optional): Target loudness in LUFS.

    Returns:
        np.ndarray: Loudness-normalized audio data.
    """
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)

    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, new_loudness)
    return loudness_normalized_audio


def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Butterworth bandpass filter.

    Args:
        lowcut (float): Low cut-off frequency.
        highcut (float): High cut-off frequency.
        fs (int): Sample rate.
        order (int, optional): Filter order.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filter coefficients.
    """
    nyq = 0.5 * fs 
    low = lowcut / nyq 
    high = highcut / nyq 
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
    """
    Apply a bandpass filter to an audio signal.

    Args:
        data (np.ndarray): Audio data.
        lowcut (float): Low cut-off frequency.
        highcut (float): High cut-off frequency.
        fs (int): Sample rate.
        order (int, optional): Filter order.

    Returns:
        np.ndarray: Filtered audio data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return np.array(y)

def perform_cross_correlation(
    my_waveform_session: torch.Tensor, waveform_instruction: torch.Tensor, sr: int, peak_height: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform cross-correlation and identify peaks.

    Args:
        my_waveform_session (torch.Tensor): Session waveform.
        waveform_instruction (torch.Tensor): Instruction waveform.
        sr (int): Sample rate.
        peak_height (float): Peak height threshold.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Normalized cross-correlation and peak indices.
    """
    cross_correlation = correlate(
        my_waveform_session, waveform_instruction, mode="full", method="fft"
    )
    energy_session = torch.sum(my_waveform_session**2)
    energy_instruction = torch.sum(waveform_instruction**2)
    normalized_cross_correlation = cross_correlation / torch.sqrt(
        energy_session * energy_instruction
    )
    normalized_cross_correlation = np.abs(normalized_cross_correlation)
    min_max_normalized_cross_corr = min_max_normalization(normalized_cross_correlation)
    peaks_indices, _ = find_peaks(
        min_max_normalized_cross_corr, height=peak_height, distance=sr / 5
    )
    return min_max_normalized_cross_corr, peaks_indices

def get_peak_height(instruction_order: int) -> Tuple[float, float]:
    """
    Get the peak height and absolute peak height for a given instruction order.

    Args:
        instruction_order (int): Order of the instruction.

    Returns:
        Tuple[float, float]: Peak height and absolute peak height.
    """
    if instruction_order == 20:
        peak_height = Config.long_instructions_peak_height
        absolute_peak_height = Config.long_instructions_absolute_peak_height
    elif instruction_order == 37:
        peak_height = Config.long_instructions_peak_height
        absolute_peak_height = Config.long_instructions_absolute_peak_height
    elif instruction_order < 10:
        peak_height = Config.long_instructions_peak_height
        absolute_peak_height = Config.long_instructions_absolute_peak_height
    elif instruction_order < 20:
        peak_height = Config.word_instructions_peak_height
        absolute_peak_height = Config.word_instructions_absolute_peak_height
    else:
        peak_height = Config.non_word_instructions_peak_height
        absolute_peak_height = Config.non_word_instructions_absolute_peak_height
    return peak_height, absolute_peak_height


def get_session_timings(instruction_order: int, instructions_timings: List[Optional[dict]], waveform_session: torch.Tensor, sr: int) -> Tuple[int, int]:
    """
    Get the start and end times for a session based on instruction order.

    Args:
        instruction_order (int): Order of the instruction.
        instructions_timings (List[Optional[dict]]): List of instruction timings.
        waveform_session (torch.Tensor): Session waveform.
        sr (int): Sample rate.

    Returns:
        Tuple[int, int]: Start and end times for the session.
    """
    if instruction_order == 20:
        prev = 9
        while prev >= 0 and instructions_timings[prev] is None:
            prev -= 1
        if instructions_timings[prev] is not None:
            my_new_session_start = (
                math.ceil(
                    instructions_timings[prev]["start"] # type: ignore
                    + instructions_timings[prev]["duration"] # type: ignore
                )
                * sr
            )
        else:
            my_new_session_start = 0
        my_new_session_end = waveform_session.shape[0]

    elif instruction_order == 37:
        prev = 20

        while prev >= 0 and instructions_timings[prev] is None:
            prev -= 1

        if instructions_timings[prev] is not None:
            my_new_session_start = (
                math.ceil(
                    instructions_timings[prev]["start"] # type: ignore
                    + instructions_timings[prev]["duration"] # type: ignore
                )
                * sr
            )
        else:
            my_new_session_start = 0
        my_new_session_end = waveform_session.shape[0]
    else:
        if instruction_order > 0:
            i = 1
            while (
                instruction_order - i >= 0
                and instructions_timings[instruction_order - i] is None
            ):
                i += 1
            if instruction_order - i < 0:
                my_new_session_start = 0
            else:
                my_new_session_start = (
                    math.ceil(
                        instructions_timings[instruction_order - i]["start"] # type: ignore
                        + instructions_timings[instruction_order - i]["duration"] # type: ignore
                    )
                    * sr
                )
        else:
            my_new_session_start = 0

        if instruction_order > 9 and instruction_order < 20:
            if instructions_timings[20] is not None:
                my_new_session_end = int(instructions_timings[20]["start"]) * sr
            else:
                my_new_session_end = waveform_session.shape[0]
        elif instruction_order > 20 and instruction_order < 37:
            if instructions_timings[37] is not None:
                my_new_session_end = int(instructions_timings[37]["start"]) * sr
            else:
                my_new_session_end = waveform_session.shape[0]
        else:
            my_new_session_end = waveform_session.shape[0]

    return my_new_session_start, my_new_session_end

def process_instruction_files(
    instruction_file_paths: List[str],
    waveform_session: torch.Tensor,
    sr: int,
    session_file_path: str,
    cross_correlations_folder: str,
) -> List[Optional[dict]]:
    """
    Process instruction files and align them with session data, using the specified correlations folder for plots.

    Args:
        instruction_file_paths (List[str]): Paths to instruction files.
        waveform_session (torch.Tensor): Session waveform.
        sr (int): Sample rate.
        session_file_path (str): Path to the session file.
        cross_correlations_folder (str): Folder to save cross-correlation plots.

    Returns:
        List[Optional[dict]]: List of instruction timings.
    """
    instructions_timings: List[Optional[dict]] = [None] * len(instruction_file_paths)
    for instruction_file_path in instruction_file_paths:
        waveform_instruction, sr, instruction_duration = load_audio(
            instruction_file_path,
            new_sample_rate=Config.new_sample_rate,
            filtering=Config.filtering,
            lowcut=Config.lowcut,
            highcut=Config.highcut,
            normalization=Config.normalization,
            max_length=Config.seconds_threshold,
        )
        instruction_order = int(os.path.basename(instruction_file_path)[:2])
        peak_height, absolute_peak_height = get_peak_height(instruction_order)
        my_new_session_start, my_new_session_end = get_session_timings(
            instruction_order, instructions_timings, waveform_session, sr
        )
        my_waveform_session = waveform_session[my_new_session_start:my_new_session_end]
        if my_waveform_session.shape[0] == 0:
            continue

        cross_correlation, peaks_indices = perform_cross_correlation(
            my_waveform_session, waveform_instruction, sr, peak_height
        )
        update_instruction_timings(
            instructions_timings,
            instruction_order,
            peaks_indices,
            cross_correlation,
            waveform_instruction,
            sr,
            my_new_session_start,
            instruction_duration,
            absolute_peak_height,
            instruction_file_path,
        )

        plot_cross_correlation(
            session_file_path,
            instruction_file_path,
            cross_correlation,
            peaks_indices.tolist(),
            cross_correlations_folder,
            sr,
            my_new_session_start,
        )
    return instructions_timings

def update_instruction_timings(
    instructions_timings: List[Optional[dict]],
    instruction_order: int,
    peaks_indices: np.ndarray,
    normalized_cross_correlation: np.ndarray,
    waveform_instruction: torch.Tensor,
    sr: int,
    my_new_session_start: int,
    instruction_duration: float,
    absolute_peak_height: float,
    instruction_file_path: str,
) -> None:
    """
    Update the instruction timings based on cross-correlation peaks.

    Args:
        instructions_timings (List[Optional[dict]]): List of instruction timings.
        instruction_order (int): Order of the instruction.
        peaks_indices (np.ndarray): Indices of the peaks.
        normalized_cross_correlation (np.ndarray): Normalized cross-correlation values.
        waveform_instruction (torch.Tensor): Instruction waveform.
        sr (int): Sample rate.
        my_new_session_start (int): Start time of the new session.
        instruction_duration (float): Duration of the instruction.
        absolute_peak_height (float): Absolute peak height threshold.
        instruction_file_path (str): Path to the instruction file.

    Returns:
        None
    """
    if (
        len(peaks_indices) == 1
        and max(normalized_cross_correlation) > absolute_peak_height
    ):
        max_corr_index = peaks_indices[0]
        shift = (max_corr_index - (len(waveform_instruction) - 1)) / sr
        instructions_timings[instruction_order] = {
            "instruction_order": instruction_order,
            "start": shift + my_new_session_start / sr,
            "duration": instruction_duration,
            "label": os.path.basename(instruction_file_path)[:-4],
        }