#!/usr/bin/env python
# coding: utf-8

# Imports
import sys
import torch
import torchaudio
import os
from scipy.signal import correlate
import numpy as np
from tqdm import tqdm
import warnings
from scipy.signal import find_peaks
from transformers import pipeline
import string
import json
import argparse
import whisperx
from datetime import datetime
from config import Config
from utility.utility import generate_json, preprocess_and_tokenize, load_audio, get_peak_height, get_session_timings, get_instructions, plot_cross_correlation, save_audacity_file, find_story_in_session, min_max_normalization

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup environment
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

def parse_args():
    """ Parse command line arguments including hyperparameters. """
    parser = argparse.ArgumentParser(description="Experiment script for session processing.")
    parser.add_argument("session_name", help="Name of the session")
    parser.add_argument("transcript_tool", help="Name of the transcript tool")
    parser.add_argument("new_sample_rate", type=int, default=8000, help="New sample rate for audio processing")
    parser.add_argument("highcut", type=int, default=3500, help="Highcut frequency for filtering")
    parser.add_argument("lowcut", type=int, default=500, help="Lowcut frequency for filtering")
    parser.add_argument("normalization", type=bool, default=True, help="Enable or disable normalization")
    parser.add_argument("filtering", type=bool, default=True, help="Enable or disable filtering")
    parser.add_argument("seconds_threshold", type=float, default=3.0, help="Threshold in seconds for audio processing")
    parser.add_argument("story_absolute_peak_height", type=float, default=0.65, help="Absolute peak height for story detection")
    parser.add_argument("long_instructions_peak_height", type=float, default=0.6, help="Peak height for long instructions")
    parser.add_argument("word_instructions_peak_height", type=float, default=0.8, help="Peak height for word instructions")
    parser.add_argument("non_word_instructions_peak_height", type=float, default=0.8, help="Peak height for non-word instructions")
    parser.add_argument("long_instructions_absolute_peak_height", type=float, default=0.01, help="Absolute peak height for long instructions")
    parser.add_argument("word_instructions_absolute_peak_height", type=float, default=0.05, help="Absolute peak height for word instructions")
    parser.add_argument("non_word_instructions_absolute_peak_height", type=float, default=0.05, help="Absolute peak height for non-word instructions")
    return parser.parse_args()

def setup_directories(base_dir, config):
    """Set up directories based on the current date and time and save config to a JSON file."""
    # Create a folder with the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_folder = os.path.join(base_dir, f"data_run_{current_time}")
    audacity_folder = os.path.join(data_folder, "audacity")
    transcriptions_folder = os.path.join(data_folder, "transcriptions")
    similarity_folder = os.path.join(data_folder, "similarity")
    cross_correlations_folder = os.path.join(data_folder, "cross_correlation")

    # Create directories
    os.makedirs(audacity_folder, exist_ok=True)
    os.makedirs(transcriptions_folder, exist_ok=True)
    os.makedirs(similarity_folder, exist_ok=True)
    os.makedirs(cross_correlations_folder, exist_ok=True)

    # Save configuration to a JSON file within the data folder
    config_path = os.path.join(data_folder, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return audacity_folder, transcriptions_folder, similarity_folder, cross_correlations_folder

def load_or_transcribe_audio(session_file_path, transcript_tool, transcriptions_folder):
    """ Load or transcribe audio based on the availability of a transcription. """
    session_transcript_file = os.path.join(transcriptions_folder, f"{os.path.basename(session_file_path)[:-4]}.json")
    if os.path.exists(session_transcript_file):
        with open(session_transcript_file, "r", encoding='utf-8') as f:
            result = json.load(f)
    else:
        result = transcribe_audio(session_file_path, transcript_tool)
        os.makedirs(transcriptions_folder, exist_ok=True)
        with open(session_transcript_file, "w", encoding='utf-8') as f:
            json.dump(result, f)
    return result

def transcribe_audio(session_file_path, transcript_tool):
    """ Transcribe audio using Whisper or WhisperX based on the transcript tool specified. """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if transcript_tool == "whisper":
        return transcribe_with_whisper(session_file_path, device)
    else:
        return transcribe_with_whisperx(session_file_path, device)

def transcribe_with_whisper(session_file_path, device):
    """ Use Whisper for transcription. """
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps='word',
        torch_dtype=torch_dtype,
        device=device,
    )
    waveform, _ = torchaudio.load(session_file_path)
    result = pipe(waveform.numpy(), generate_kwargs={"language": "english"})
    return result

def transcribe_with_whisperx(session_file_path, device):
    """ Use WhisperX for transcription and format output. """
    whisperx_model = whisperx.load_model("tiny", device, compute_type="int8", language="en")
    audio = whisperx.load_audio(session_file_path)
    result = whisperx_model.transcribe(audio, language="en", batch_size=16)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    formatted_result = generate_json(aligned_result)
    return formatted_result

def process_transcripts(result):
    """ Process raw transcript data into a structured format. """
    if 'chunks' not in result:
        return []  # Return an empty list or handle as needed
    chunks = result['chunks']
    session_word_by_word = []
    for chunk in chunks:
        session_word_by_word.append({
            'text': chunk['text'].translate(str.maketrans('', '', string.punctuation)) if 'text' in chunk else None,
            'start': chunk['timestamp'][0] if 'timestamp' in chunk else None,
            'end': chunk['timestamp'][1] if 'timestamp' in chunk else None
        })
    return session_word_by_word

def tokenize_session_data(session_word_by_word):
    """ Tokenize the text from the session transcripts. """
    session_tokens = [preprocess_and_tokenize(item['text']) for item in session_word_by_word]
    session_tokens_flat = [item for sublist in session_tokens for item in sublist]
    return session_tokens_flat

def find_stories_in_session(session_word_by_word, session_tokens_flat, similarity_folder, session_file_path, sr, story_absolute_peak_height):
    """ Find and time stories within the session data without adjusting the waveform session. """
    story_timings = [None] * len(Config.stories)
    stories_starts = []
    for story_index, story in tqdm(enumerate(Config.stories), desc='Stories'):
        story_tokens = [preprocess_and_tokenize(item) for item in story.split(' ')]
        story_tokens_flat = [item for sublist in story_tokens for item in sublist]

        similarity_session_folder = os.path.join(similarity_folder, os.path.basename(session_file_path)[:-4])
        os.makedirs(similarity_session_folder, exist_ok=True)
        similarity_figure_file_name = f"story_{story_index}"

        story_start = find_story_in_session(session_word_by_word, session_tokens_flat, story_tokens_flat, story_absolute_peak_height,
                                           file_path=os.path.join(similarity_session_folder, f"{similarity_figure_file_name}.png"))
        if story_start is not None:
            story_timings[story_index] = {
                "instruction_order": story_index,
                "start": story_start,
                "duration": 0,
                "label": f"story_{story_index}"
            }
            stories_starts.append(story_start)

    return story_timings, stories_starts

def process_instruction_files(instruction_file_paths, waveform_session, sr, session_file_path, cross_correlations_folder):
    """ Process instruction files and align them with session data, using the specified correlations folder for plots. """
    instructions_timings = [None] * len(instruction_file_paths)
    for instruction_file_path in tqdm(instruction_file_paths, desc='Instruction files'):
        tqdm.write(f"Processing {instruction_file_path}")
        waveform_instruction, sr, instruction_duration = load_audio(instruction_file_path, new_sample_rate=Config.new_sample_rate,
                                                                    filtering=Config.filtering, lowcut=Config.lowcut,
                                                                    highcut=Config.highcut, normalization=Config.normalization,
                                                                    max_length=Config.seconds_threshold)
        instruction_order = int(os.path.basename(instruction_file_path)[:2])
        peak_height, absolute_peak_height = get_peak_height(instruction_order)
        my_new_session_start, my_new_session_end = get_session_timings(instruction_order, instructions_timings, waveform_session, sr)
        my_waveform_session = waveform_session[my_new_session_start:my_new_session_end]
        if my_waveform_session.shape[0] == 0:
            continue

        cross_correlation, peaks_indices = perform_cross_correlation(my_waveform_session, waveform_instruction, sr, peak_height)
        update_instruction_timings(instructions_timings, instruction_order, peaks_indices, cross_correlation, waveform_instruction, sr, my_new_session_start, instruction_duration, absolute_peak_height, instruction_file_path)

        # Now with updated time conversion
        plot_cross_correlation(
            session_file_path, 
            instruction_file_path, 
            cross_correlation, 
            peaks_indices, 
            cross_correlations_folder, 
            sr, 
            my_new_session_start
        )
    return instructions_timings

def perform_cross_correlation(my_waveform_session, waveform_instruction, sr, peak_height):
    """ Perform cross-correlation and identify peaks. """
    cross_correlation = correlate(my_waveform_session, waveform_instruction, mode='full', method='fft')
    energy_session = torch.sum(my_waveform_session ** 2)
    energy_instruction = torch.sum(waveform_instruction ** 2)
    normalized_cross_correlation = cross_correlation / torch.sqrt(energy_session * energy_instruction)
    normalized_cross_correlation = np.abs(normalized_cross_correlation.numpy())
    min_max_normalized_cross_corr = min_max_normalization(normalized_cross_correlation)
    peaks_indices, _ = find_peaks(min_max_normalized_cross_corr, height=peak_height, distance=sr / 5)
    return min_max_normalized_cross_corr, peaks_indices

def update_instruction_timings(instructions_timings, instruction_order, peaks_indices, normalized_cross_correlation, waveform_instruction, sr, my_new_session_start, instruction_duration, absolute_peak_height, instruction_file_path):
    """ Update the instruction timings based on cross-correlation peaks. """
    if len(peaks_indices) == 1 and max(normalized_cross_correlation) > absolute_peak_height:
        max_corr_index = peaks_indices[0]
        shift = (max_corr_index - (len(waveform_instruction) - 1)) / sr
        instructions_timings[instruction_order] = {
            "instruction_order": instruction_order,
            "start": shift + my_new_session_start / sr,
            "duration": instruction_duration,
            "label": os.path.basename(instruction_file_path)[:-4]
        }

def finalize_results(audacity_folder, session_file_path, story_timings, instructions_timings):
    """ Save the final timings to an Audacity label file. """
    all_timings = instructions_timings + story_timings
    save_audacity_file(audacity_folder, session_file_path, all_timings)

def main():
    args = parse_args()
    config = {
        'new_sample_rate': args.new_sample_rate,
        'highcut': args.highcut,
        'lowcut': args.lowcut,
        'normalization': args.normalization,
        'filtering': args.filtering,
        'seconds_threshold': args.seconds_threshold,
        'story_absolute_peak_height': args.story_absolute_peak_height,
        'long_instructions_peak_height': args.long_instructions_peak_height,
        'word_instructions_peak_height': args.word_instructions_peak_height,
        'non_word_instructions_peak_height': args.non_word_instructions_peak_height,
        'long_instructions_absolute_peak_height': args.long_instructions_absolute_peak_height,
        'word_instructions_absolute_peak_height': args.word_instructions_absolute_peak_height,
        'non_word_instructions_absolute_peak_height': args.non_word_instructions_absolute_peak_height
    }
    audacity_folder, transcriptions_folder, similarity_folder, cross_correlations_folder = setup_directories("data", config)    
    print("Session Name:", args.session_name)
    session_file_path = os.path.join(Config.sessions_folder, args.session_name)

    result = load_or_transcribe_audio(session_file_path, args.transcript_tool, transcriptions_folder)
    session_word_by_word = process_transcripts(result)
    session_tokens_flat = tokenize_session_data(session_word_by_word)

    waveform_session, sr, _ = load_audio(session_file_path, new_sample_rate=args.new_sample_rate, filtering=args.filtering, lowcut=args.lowcut, highcut=args.highcut, normalization=args.normalization)
    story_timings, stories_starts = find_stories_in_session(session_word_by_word, session_tokens_flat, similarity_folder, session_file_path, sr, args.story_absolute_peak_height)
    instruction_file_paths = get_instructions(Config.instructions_folder)
    instructions_timings = process_instruction_files(instruction_file_paths, waveform_session, sr, session_file_path, cross_correlations_folder)
    finalize_results(audacity_folder, session_file_path, story_timings, instructions_timings)

if __name__ == "__main__":
    main()