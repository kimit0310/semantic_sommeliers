# Imports
import argparse
import json
import logging
import math
import os
import string
import sys
import csv
import glob
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import pytorch_lightning as pl
import torch
import torchaudio
import torchaudio.transforms as T
import whisperx
from scipy.signal import butter, correlate, find_peaks, lfilter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import WhisperTimeStampLogitsProcessor, pipeline

from src.config import Config

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pytorch_lightning')
warnings.filterwarnings("ignore", category=UserWarning, module='pyannote')
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
logging.getLogger("torch").setLevel(logging.ERROR)
pl.utilities.rank_zero_only.rank_zero_warn = lambda *args, **kwargs: None

# Assuming config.py is in the project's root directory, similar to experiments.py
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)




def load_audio(
    path,
    new_sample_rate=None,
    filtering=False,
    lowcut=None,
    highcut=None,
    normalization=False,
    max_length=None,
):
    waveform, sample_rate = torchaudio.load(path)
    instruction_duration = waveform.shape[1] / sample_rate

    if new_sample_rate:
        # TODO: add filtering here to reduce aliasing. This may increase the effectiveness of our cross-correlation approach
        # Try low-pass filter in resampling to reduce aliasing, see if it has effect (rolloff https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#mfcc)
        resampler = T.Resample(sample_rate, new_sample_rate, dtype=waveform.dtype)
        waveform = resampler(waveform)
        sample_rate = new_sample_rate
    if filtering:
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


def normalize_volume(data, rate, new_loudness=-23):
    # measure the loudness first
    meter = pyln.Meter(rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to new_loudness dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, new_loudness)
    return loudness_normalized_audio


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq  # Normalize lowcut
    high = highcut / nyq  # Normalize highcut
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y











def get_peak_height(instruction_order):
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


def get_session_timings(instruction_order, instructions_timings, waveform_session, sr):
    if instruction_order == 20:
        prev = 9
        while prev >= 0 and instructions_timings[prev] is None:
            prev -= 1
        if instructions_timings[prev] is not None:
            my_new_session_start = (
                math.ceil(
                    instructions_timings[prev]["start"]
                    + instructions_timings[prev]["duration"]
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
                    instructions_timings[prev]["start"]
                    + instructions_timings[prev]["duration"]
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
                        instructions_timings[instruction_order - i]["start"]
                        + instructions_timings[instruction_order - i]["duration"]
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


def extract_sentence(data, start_time, end_time):
    """
    Extracts and returns a sentence from the given data within the specified time range.
    """
    sentence = ""
    for item in data:
        # Check if the current item's time overlaps with the given time range
        if (
            "start" in item
            and "end" in item
            and item["start"] is not None
            and item["end"] is not None
            and item["start"] <= end_time
            and item["end"] >= start_time
        ):
            sentence += item["text"]
    return sentence.strip()


def preprocess_and_tokenize(text):
    # Remove punctuation and convert to lowercase
    clean_text = "".join(
        char.lower() if char.isalnum() or char.isspace() else " " for char in text
    )
    # Tokenize by splitting on whitespace
    tokens = clean_text.split()
    return tokens


def extract_semantic_embeddings(model, sentence):
    return model.encode(sentence)


def compute_cosine_similarity(embeddings1, embeddings2):
    cosine_sim = cosine_similarity(
        embeddings1.reshape(1, -1), embeddings2.reshape(1, -1)
    )
    return np.mean(cosine_sim)


def find_story_in_session(
    session_transcript,
    session_tokens,
    story_tokens,
    story_absolute_peak_height=0.65,
    file_path=None,
):
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    times = [
        session_transcript[i]["start"]
        for i in range(len(session_tokens) - len(story_tokens) + 1)
        if i < len(session_transcript)
    ]
    similarities = []
    for i, _ in enumerate(times):
        embeddings1 = extract_semantic_embeddings(
            model, " ".join(session_tokens[i : i + len(story_tokens)])
        )
        embeddings2 = extract_semantic_embeddings(model, " ".join(story_tokens))
        similarity = compute_cosine_similarity(embeddings1, embeddings2)
        similarities.append(similarity)

    normalized_similarities = np.abs(similarities)
    if normalized_similarities.size == 0:
        return None

    min_max_normalized_similarities = min_max_normalization(normalized_similarities)
    peaks_indices, _ = find_peaks(
        min_max_normalized_similarities,
        height=story_absolute_peak_height,
        distance=len(story_tokens),
    )

    plt.figure(figsize=(10, 6))
    plt.plot(times, similarities, label="Similarity")
    plt.xlabel("Time")
    plt.ylabel("Similarity")
    plt.title("Semantic Similarity Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    
    if len(peaks_indices) > 0 and max(normalized_similarities) > story_absolute_peak_height:
        i = peaks_indices[0]
        plt.scatter(times[i], normalized_similarities[i], color="red", zorder=5)
        plt.savefig(file_path)
        return times[i]

    plt.savefig(file_path)
    return None





def load_or_transcribe_audio(
    session_file_path,
    waveform,
    sample_rate,
    transcript_tool,
    transcriptions_folder,
    last_instruction_time,
):
    """Load or transcribe audio based on the necessity of transcription."""
    session_transcript_file = os.path.join(
        transcriptions_folder, f"{os.path.basename(session_file_path)[:-4]}.json"
    )
    if not os.path.exists(session_transcript_file):
        # Calculate the starting sample for transcription based on the last instruction's timing
        start_sample = int(last_instruction_time * sample_rate)
        relevant_waveform = waveform[
            start_sample:
        ]  # Slice the waveform from last instruction end, assuming mono

        # Choose the transcription method based on the tool specified
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if transcript_tool == "whisper":
            result = transcribe_with_whisper(relevant_waveform, device)
        else:
            result = transcribe_with_whisperx(relevant_waveform, device)

        # Save transcription result
        with open(session_transcript_file, "w", encoding="utf-8") as f:
            json.dump(result, f)

    else:
        with open(session_transcript_file, "r", encoding="utf-8") as f:
            result = json.load(f)

    return result


def transcribe_with_whisper(waveform, device):
    """Transcribe a given waveform using Whisper."""
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    processor = WhisperTimeStampLogitsProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        processor=processor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=8,
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(waveform.numpy(), generate_kwargs={"language": "english"})
    return result


def transcribe_with_whisperx(waveform, device):
    """Transcribe a given waveform using WhisperX and format the output."""
    whisperx_model = whisperx.load_model(
        "large", device, compute_type="int8", language="en"
    )

    # Convert waveform to NumPy array if it's a PyTorch tensor
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    # Check if the waveform is empty and return early if it is
    if waveform.size == 0:
        print("Warning: Received empty waveform for transcription.")
        return {}  # Return an empty dict or appropriate error signal

    # Ensure the waveform is in float32 format, normalized if necessary
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32) / np.max(np.abs(waveform))

    # Transcribe using WhisperX with the prepared waveform
    result = whisperx_model.transcribe(waveform, language="en", batch_size=16)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        waveform,
        device,
        return_char_alignments=False,
    )
    formatted_result = generate_json(aligned_result)
    return formatted_result


def process_transcripts(result):
    """Process raw transcript data into a structured format."""
    if "chunks" not in result:
        return []  # Return an empty list or handle as needed
    chunks = result["chunks"]
    session_word_by_word = []
    for chunk in chunks:
        session_word_by_word.append(
            {
                "text": chunk["text"].translate(
                    str.maketrans("", "", string.punctuation)
                )
                if "text" in chunk
                else None,
                "start": chunk["timestamp"][0] if "timestamp" in chunk else None,
                "end": chunk["timestamp"][1] if "timestamp" in chunk else None,
            }
        )
    return session_word_by_word


def tokenize_session_data(session_word_by_word):
    """Tokenize the text from the session transcripts."""
    session_tokens = [
        preprocess_and_tokenize(item["text"]) for item in session_word_by_word
    ]
    session_tokens_flat = [item for sublist in session_tokens for item in sublist]
    return session_tokens_flat


def find_stories_in_session(
    session_word_by_word,
    session_tokens_flat,
    similarity_folder,
    session_file_path,
    sr,
    story_absolute_peak_height,
):
    """Find and time stories within the session data without adjusting the waveform session."""
    story_timings = [None] * len(Config.stories)
    stories_starts = []
    for story_index, story in enumerate(Config.stories):
        story_tokens = [preprocess_and_tokenize(item) for item in story.split(" ")]
        story_tokens_flat = [item for sublist in story_tokens for item in sublist]

        similarity_session_folder = os.path.join(
            similarity_folder, os.path.basename(session_file_path)[:-4]
        )
        os.makedirs(similarity_session_folder, exist_ok=True)
        similarity_figure_file_name = f"story_{story_index}"

        story_start = find_story_in_session(
            session_word_by_word,
            session_tokens_flat,
            story_tokens_flat,
            story_absolute_peak_height,
            file_path=os.path.join(
                similarity_session_folder, f"{similarity_figure_file_name}.png"
            ),
        )
        if story_start is not None:
            story_timings[story_index] = {
                "instruction_order": story_index,
                "start": story_start,
                "duration": 0,
                "label": f"story_{story_index}",
            }
            stories_starts.append(story_start)

    return story_timings, stories_starts


def process_instruction_files(
    instruction_file_paths,
    waveform_session,
    sr,
    session_file_path,
    cross_correlations_folder,
):
    """Process instruction files and align them with session data, using the specified correlations folder for plots."""
    instructions_timings = [None] * len(instruction_file_paths)
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
            peaks_indices,
            cross_correlations_folder,
            sr,
            my_new_session_start,
        )
    return instructions_timings


def perform_cross_correlation(
    my_waveform_session, waveform_instruction, sr, peak_height
):
    """Perform cross-correlation and identify peaks."""
    cross_correlation = correlate(
        my_waveform_session, waveform_instruction, mode="full", method="fft"
    )
    energy_session = torch.sum(my_waveform_session**2)
    energy_instruction = torch.sum(waveform_instruction**2)
    normalized_cross_correlation = cross_correlation / torch.sqrt(
        energy_session * energy_instruction
    )
    normalized_cross_correlation = np.abs(normalized_cross_correlation.numpy())
    min_max_normalized_cross_corr = min_max_normalization(normalized_cross_correlation)
    peaks_indices, _ = find_peaks(
        min_max_normalized_cross_corr, height=peak_height, distance=sr / 5
    )
    return min_max_normalized_cross_corr, peaks_indices


def update_instruction_timings(
    instructions_timings,
    instruction_order,
    peaks_indices,
    normalized_cross_correlation,
    waveform_instruction,
    sr,
    my_new_session_start,
    instruction_duration,
    absolute_peak_height,
    instruction_file_path,
):
    """Update the instruction timings based on cross-correlation peaks."""
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


