# Imports
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import pyloudnorm as pyln
from scipy.signal import butter, lfilter
import numpy as np
from tqdm import tqdm
import warnings
from scipy.signal import find_peaks
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import math
import argparse
import whisperx

def generate_json(input_json):
    segments = input_json["segments"]
    concatenated_text = ""
    chunks = []

    for segment in segments:
        for word_info in segment["words"]:
            concatenated_text += word_info["word"] + " "
            chunks.append({
                "text": word_info["word"],
                "timestamp": [word_info["start"], word_info["end"]]
            })

    concatenated_text = concatenated_text.strip()

    output_json = {
        "text": concatenated_text,
        "chunks": chunks
    }

    return output_json

def load_audio(path, new_sample_rate=None, filtering=False, lowcut=None, highcut=None, normalization=False, max_length=None):
    waveform, sample_rate = torchaudio.load(path)
    instruction_duration = waveform.shape[1] / sample_rate

    if new_sample_rate:
        # TODO: add filtering here to reduce aliasing. This may increase the effectiveness of our cross-correlation approach
        resampler = T.Resample(sample_rate, new_sample_rate, dtype=waveform.dtype)
        waveform = resampler(waveform)
        sample_rate = new_sample_rate
    if filtering:
        waveform = torch.tensor(bandpass_filter(waveform, lowcut=lowcut, highcut=highcut, fs=sample_rate))
    if normalization:
        waveform = torch.tensor(normalize_volume(waveform[0].numpy(), sample_rate).reshape(1, -1))
    waveform = waveform.squeeze()
    if max_length is not None:
        waveform = waveform[:int(max_length*sample_rate)]
    return waveform, sample_rate, instruction_duration

def normalize_volume(data, rate, new_loudness=-23):   
    # measure the loudness first 
    meter = pyln.Meter(rate) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)
    
    # loudness normalize audio to new_loudness dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, new_loudness)
    return loudness_normalized_audio

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq  # Normalize lowcut
    high = highcut / nyq  # Normalize highcut
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def min_max_normalization(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr

def save_cross_correlation(cross_correlation, peaks_indices, file_path):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cross_correlation)), cross_correlation, label='Cross-Correlation')
    plt.plot(peaks_indices, cross_correlation[peaks_indices], 'o', label='Top Peaks')
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    plt.title('Cross-Correlation between Session and Instruction')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)

def save_audacity_file(audacity_folder, session_file_path, instructions_timings):
    os.makedirs(audacity_folder, exist_ok=True)
    with open(os.path.join(audacity_folder, os.path.basename(session_file_path)[:-4] + '.txt'), 'w', encoding='utf-8') as file:
        for instructions_timing in instructions_timings:
            if instructions_timing is not None:
                file.write(f"{instructions_timing['start']}\t{instructions_timing['start'] + instructions_timing['duration']}\t{instructions_timing['label']}\n")
    return

def get_instructions(instructions_folder):
    instruction_file_paths = sorted([os.path.join(instructions_folder, instruction_filename) for instruction_filename in os.listdir(instructions_folder) if instruction_filename.endswith(".wav")])
    first_10_instructions = instruction_file_paths[:10]
    twenty_first_instruction = instruction_file_paths[20:21]
    last_instruction = instruction_file_paths[-1:]
    remaining_instructions = instruction_file_paths[10:20] + instruction_file_paths[21:38]
    instruction_file_paths = first_10_instructions + twenty_first_instruction + last_instruction + remaining_instructions
    return instruction_file_paths

def get_sessions(sessions_folder):
    session_file_paths = sorted([os.path.join(sessions_folder, session_filename) for session_filename in os.listdir(sessions_folder) if session_filename.endswith(".wav")])
    return session_file_paths

def plot_cross_correlation(session_file_path, instruction_file_path, standardized_normalized_cross_corr, peaks_indices):
    # TODO: PLOT CROSS CORRELATION WITH ABSOLUTE VALUES
    os.makedirs(cross_correlations_folder, exist_ok=True)
    cross_correlations_session_folder = os.path.join(cross_correlations_folder, os.path.basename(session_file_path)[:-4])
    os.makedirs(cross_correlations_session_folder, exist_ok=True)
    cross_correlation_figure_file_name = f"{os.path.basename(instruction_file_path)[:-4]}_{os.path.basename(session_file_path)[:-4]}"
    save_cross_correlation(standardized_normalized_cross_corr, 
                            peaks_indices, 
                            os.path.join(cross_correlations_session_folder, f"{cross_correlation_figure_file_name}.png"))        

def get_peak_height(instruction_order):
    if instruction_order == 20:
        peak_height = long_instructions_peak_height
        absolute_peak_height = long_instructions_absolute_peak_height
    elif instruction_order == 37:
        peak_height = long_instructions_peak_height
        absolute_peak_height = long_instructions_absolute_peak_height
    elif instruction_order < 10:
        peak_height = long_instructions_peak_height
        absolute_peak_height = long_instructions_absolute_peak_height
    elif instruction_order < 20:
        peak_height = word_instructions_peak_height
        absolute_peak_height = word_instructions_absolute_peak_height
    else:
        peak_height = non_word_instructions_peak_height
        absolute_peak_height = non_word_instructions_absolute_peak_height
    return peak_height, absolute_peak_height

def get_session_timings(instruction_order, instructions_timings, waveform_session, sr):
    if instruction_order == 20:
        prev = 9
        while prev >= 0 and instructions_timings[prev] is None:
            prev -= 1
        if instructions_timings[prev] is not None:
            my_new_session_start = math.ceil(instructions_timings[prev]['start'] + instructions_timings[prev]['duration']) * sr
        else:
            my_new_session_start = 0
        my_new_session_end = waveform_session.shape[0]

    elif instruction_order == 37:
        prev = 20
        
        while prev >= 0 and instructions_timings[prev] is None:
            prev -= 1

        if instructions_timings[prev] is not None:
            my_new_session_start = math.ceil(instructions_timings[prev]['start'] + instructions_timings[prev]['duration']) * sr
        else:
            my_new_session_start = 0
        my_new_session_end = waveform_session.shape[0]
    else:
        if instruction_order > 0:
            i = 1
            while instruction_order - i >= 0 and instructions_timings[instruction_order - i] is None:
                i += 1
            if instruction_order-i < 0:
                my_new_session_start = 0
            else:
                my_new_session_start = math.ceil(instructions_timings[instruction_order - i]['start'] + instructions_timings[instruction_order - i]['duration']) * sr
        else:
            my_new_session_start = 0

        if instruction_order > 9 and instruction_order < 20:
            if instructions_timings[20] is not None:
                my_new_session_end = int(instructions_timings[20]['start']) * sr
            else:
                my_new_session_end = waveform_session.shape[0]
        elif instruction_order > 20 and instruction_order < 37:
            if instructions_timings[37] is not None:
                my_new_session_end = int(instructions_timings[37]['start']) * sr
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
        if 'start' in item and 'end' in item and item['start'] is not None and item['end'] is not None and item['start'] <= end_time and item['end'] >= start_time:
            sentence += item['text']
    return sentence.strip()

def preprocess_and_tokenize(text):
    # Remove punctuation and convert to lowercase
    clean_text = ''.join(char.lower() if char.isalnum() or char.isspace() else ' ' for char in text)
    # Tokenize by splitting on whitespace
    tokens = clean_text.split()
    return tokens

def extract_semantic_embeddings(model, sentence):
    return model.encode(sentence)

def compute_cosine_similarity(embeddings1, embeddings2):
    cosine_sim = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1))
    return np.mean(cosine_sim)

def find_story_in_session(session_transcript, session_tokens, story_tokens, threshold=0.5, file_path=None):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Search for Match and Track Timestamps
    times = [session_transcript[i]['start'] for i in range(len(session_tokens) - len(story_tokens) + 1) if i < len(session_transcript)]
    similarities = []
    for i, _ in enumerate(times):
        embeddings1 = extract_semantic_embeddings(model, ' '.join(session_tokens[i:i+len(story_tokens)]))
        embeddings2 = extract_semantic_embeddings(model, ' '.join(story_tokens))
        similarity = compute_cosine_similarity(embeddings1, embeddings2)
        similarities.append(similarity)

    normalized_similarities = np.abs(similarities)
    min_max_normalized_similarities = min_max_normalization(normalized_similarities)
    peaks_indices, _ = find_peaks(min_max_normalized_similarities, height=threshold, distance=len(story_tokens))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(times, similarities, label='Similarity')
    plt.xlabel('Time')
    plt.ylabel('Similarity')
    plt.title('Semantic Similarity Over Time')
    plt.xticks(rotation=45) # Rotating x-axis labels for better readability

    absolute_peak_height = 0.75
    print("max(normalized_similarities)")
    print(max(normalized_similarities))
    print("len(peaks_indices)")
    print(len(peaks_indices))
    

    if len(peaks_indices) == 1:
        if max(normalized_similarities) > absolute_peak_height: 
            i = peaks_indices[0]
            max_similarity = normalized_similarities[i]        
            plt.scatter(times[i], max_similarity, color='red', zorder=5)

    plt.tight_layout() # Adjust layout to not cut off labels
    plt.legend()
    plt.savefig(file_path)

    if len(peaks_indices) == 1:
        if max(normalized_similarities) > absolute_peak_height:
            pass
        else:
            return None
    else:
        return None

    start_time = session_transcript[i]['start']
    return start_time