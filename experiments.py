#!/usr/bin/env python
# coding: utf-8

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

warnings.filterwarnings("ignore")

data_folder = "data/"
instructions_folder = os.path.join(data_folder, "instructions")
sessions_folder = os.path.join(data_folder, "sessions")
cross_correlations_folder = os.path.join(data_folder, "cross_correlation")
audacity_folder = os.path.join(data_folder, "audacity")
transcriptions_folder = os.path.join(data_folder, "transcriptions")
similarity_folder = os.path.join(data_folder, "similarity")

############## PARAMS ##############
normalization = True
filtering = True
new_sample_rate = 8000
lowcut = 300.0
highcut = 3000.0
seconds_threshold = 3.0
long_instructions_peak_height = 0.6
word_instructions_peak_height = 0.8
non_word_instructions_peak_height = 0.8
long_instructions_absolute_peak_height = 0.01
word_instructions_absolute_peak_height = 0.05
non_word_instructions_absolute_peak_height = 0.05
story1 = 'It was the first day of school. It was a tough day for all the kids. One girl had a really hard time because nobody could say her name. Her name was Peggy Babcock. Go ahead. Try and say it three times quickly. “Peggy Babcock Peggy Babcock Peggy Babcock.” Not easy going, right? She was afraid to say hello to any of the other kids on the playground. One boy walked up to her and asked what her name was. She said “When you hear my name it sounds simple but no one can say it. It is Peggy Babcock.” He laughed and said “Your name is tricky but mine is better. It sounds simple but no one can remember it. It is Jonas Norvin Sven Arthur Schwinn Bart Winston Ulysses M.” Peggy laughed and said “Easy. Your name sounds like Joan is nervous when others win. But you win some, you lose some. How do you like my version?" Jonas was so happy that he said “Lets be friends. I will call you PB.” The pair of them stuck so close to each other that everyone at school called them “PB and J.”'
story2 = 'Some time ago, in a place neither near nor far, there lived a king who did not know how to count, not even to zero. Some say this is the reason he would always wish for more — more food, more gold, more land. He simply did not realize how much he already owned. Everyone in his kingdom could do the math and tally bushels of corn, loaves of bread, and urns of gold. But how would they measure the height of his castle or the stretch of his kingdom? You might think “Aaah, ooh, easy — just measure it in meters!” But in those days, the useless unit of measure was based on stains splattered along the king s cloak while drinking shrub juice. The kingdom needed a new way of counting distance. “A kingdom without a proper ruler,” proclaimed the king, “is like riches without measure.” He launched a challenge amid trumpets, drums, flags and cannons. “The person who creates a unit of measure fit for a ruler will be rewarded beyond measure!” A tall order indeed! The first person to come forward was a bulky locksmith with a stiff jaw. He approached the king with an air of secrecy and whispered, “I have the key to measure the kingdom, but only I can wield it.” He then rubbed his beard and pulled the key from his locks of oily hair. The key turned out to be a hair itself! “Judge the reach of my vast kingdom with a hair s width?” laughed the king. “What a poor idea. That would take forever or longer!” The second person eager for the prize was a fidgety boy who knew all numbers (including zero). He produced a curious object from one of his many pockets. It was a complex shape that seemed to change proportions depending on which direction you gazed upon it. The boy said in a measured voice, “This polyhedron has many edges, with each edge of a different length. Only a king could be counted on to use it justly.” He gave the king an awful earful of an explanation that went on and on. The long and the short of it was that the king could make no more use of it than of a puddle of spilled oatmeal. Finally, a little girl with a big idea tugged on the mismeasured cloak of the king. The king sized up the little girl with the big idea and said “I don’t have time for this, and for that matter, I have no concept of space, either.” The girl looked up, then down, then spun around and blurted out: “Aren’t you able to solve the puzzle yourself? Why must you break up your kingdom into tiny pieces when everything around you is Humpty Dumpty together again? Your kingdom IS a unit and you are the ruler.” The king — startled, befuddled, and bemused — found the words wise. He aimed to be satisfied with all around him, big or small or somewhere in between.'
stories = [story1, story2]
session_file_path = os.path.join(sessions_folder, "5023141_speech_language_non.wav")
############## PARAMS ##############

def load_audio(path, new_sample_rate=None, filtering=False, lowcut=None, highcut=None, normalization=False, max_length=None):
    waveform, sample_rate = torchaudio.load(path)
    instruction_duration = waveform.shape[1] / sample_rate

    if new_sample_rate:
        # TODO: add filtering 
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
    plt.plot(range(len(cross_correlation)), cross_correlation, label='Normalized Cross-Correlation')
    plt.plot(peaks_indices, cross_correlation[peaks_indices], 'o', label='Top Peaks')
    plt.xlabel('Lag')
    plt.ylabel('Normalized Cross-Correlation')
    plt.title('Normalized Cross-Correlation between Session and Instruction')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)

def save_audacity_file(session_file_path, instructions_timings):
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

# TODO: PLOT CROSS CORRELATION WITH ABSOLUTE VALUES
def plot_cross_correlation(session_file_path, instruction_file_path, standardized_normalized_cross_corr, peaks_indices):
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
    
    Parameters:
    - data (list of dicts): The input data containing 'text', 'start', and 'end' keys.
    - start_time (float): The start time of the range.
    - end_time (float): The end time of the range.
    
    Returns:
    - str: The sentence formed by concatenating 'text' values within the time range.
    """
    sentence = ""
    for item in data:
        # Check if the current item's time overlaps with the given time range
        if 'start' in item and 'end' in item and item['start'] is not None and item['end'] is not None and item['start'] <= end_time and item['end'] >= start_time:
            sentence += item['text']
    return sentence.strip()

# Preprocessing and Tokenization
def preprocess_and_tokenize(text):
    # Remove punctuation and convert to lowercase
    clean_text = ''.join(char.lower() if char.isalnum() or char.isspace() else ' ' for char in text)
    # Tokenize by splitting on whitespace
    tokens = clean_text.split()
    return tokens

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
def extract_semantic_embeddings(model, sentence):
    return model.encode(sentence)

def compute_cosine_similarity(embeddings1, embeddings2):
    cosine_sim = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1))
    return np.mean(cosine_sim)

# Search for Match and Track Timestamps
def find_story_in_session(session_transcript, session_tokens, story_tokens, threshold=0.5, file_path=None):
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

    absolute_peak_height = 0.9
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


############################
device = "cuda" if torch.cuda.is_available() else "cpu"
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

def __main__():
    # TODO: read params from command line
    print(session_file_path)
    instruction_file_paths = get_instructions(instructions_folder)
    story_timings = [ None ] * len(stories)
    instructions_timings = [ None ] * len(instruction_file_paths)

    waveform_session, sr, _ = load_audio(session_file_path, new_sample_rate=new_sample_rate, filtering=filtering, lowcut=lowcut, highcut=highcut, normalization=normalization)
    
    session_transcript_file = os.path.join(transcriptions_folder, f"{os.path.basename(session_file_path)[:-4]}.json")
    if os.path.exists(session_transcript_file):
        with open(session_transcript_file, "r", encoding='utf-8') as f:
            result = json.load(f)
    else:
        result = pipe(waveform_session.numpy(), generate_kwargs={"language": "english"})
        with open(session_transcript_file, "w", encoding='utf-8') as f:
            json.dump(result, f)
    chunks = result['chunks']
    session_word_by_word = []
    for chunk in chunks:
        session_word_by_word.append({
            'text': chunk['text'].translate(str.maketrans('', '', string.punctuation)) if 'text' in chunk else None,
            'start': chunk['timestamp'][0] if 'timestamp' in chunk else None,
            'end': chunk['timestamp'][1] if 'timestamp' in chunk else None
        })
    
    session_tokens = [preprocess_and_tokenize(item['text']) for item in session_word_by_word]
    session_tokens_flat = [item for sublist in session_tokens for item in sublist]

    stories_starts = []
    for story_index, story in tqdm(enumerate(stories), desc='Stories'):
        story_tokens = [preprocess_and_tokenize(item) for item in story.split('.')[0].split(' ')]

        # Flatten session tokens
        story_tokens_flat = [item for sublist in story_tokens for item in sublist]

        os.makedirs(similarity_folder, exist_ok=True)
        similarity_session_folder = os.path.join(similarity_folder, os.path.basename(session_file_path)[:-4])
        os.makedirs(similarity_session_folder, exist_ok=True)
        similarity_figure_file_name = f"story_{story_index}"

        story_start = find_story_in_session(session_word_by_word, session_tokens_flat, story_tokens_flat, threshold=0.9, file_path=os.path.join(similarity_session_folder, f"{similarity_figure_file_name}.png"))
        if story_start is not None:
            story_timings[story_index] = {
                    "instruction_order": story_index,
                    "start": story_start,
                    "duration": 0, 
                    "label": f"story_{story_index}"
                }
            stories_starts.append(story_start)
    
    if len(stories_starts) > 0:
        min_stories_start = min(stories_starts)
        waveform_session = waveform_session[:int(min_stories_start*sr)]

    for instruction_file_path in tqdm(instruction_file_paths, desc='Instruction files'):
        tqdm.write(f"Processing {instruction_file_path}")
        waveform_instruction, sr, instruction_duration = load_audio(instruction_file_path, new_sample_rate=new_sample_rate, filtering=filtering, lowcut=lowcut, highcut=highcut, normalization=normalization, max_length=seconds_threshold)        
        instruction_order = int(os.path.basename(instruction_file_path)[:2])

        peak_height, absolute_peak_height = get_peak_height(instruction_order)
        my_new_session_start, my_new_session_end = get_session_timings(instruction_order, instructions_timings, waveform_session, sr)
        my_waveform_session = waveform_session[my_new_session_start:my_new_session_end]
        if my_waveform_session.shape[0] == 0:
            break

        cross_correlation = np.correlate(my_waveform_session, waveform_instruction, mode='full')

        energy_session = torch.sum(my_waveform_session ** 2)
        energy_instruction = torch.sum(waveform_instruction ** 2)

        normalized_cross_correlation = cross_correlation / torch.sqrt(energy_session * energy_instruction)
        normalized_cross_correlation = np.abs(normalized_cross_correlation.numpy())

        min_max_normalized_cross_corr = min_max_normalization(normalized_cross_correlation)
        peaks_indices, _ = find_peaks(min_max_normalized_cross_corr, height=peak_height, distance=sr/5)

        if len(peaks_indices) == 1:
            if max(normalized_cross_correlation) > absolute_peak_height:
                max_corr_index = peaks_indices[0]

                shift = (max_corr_index - (len(waveform_instruction) - 1))/ sr
                instructions_timings[instruction_order] = {
                    "instruction_order": instruction_order,
                    "start": shift + my_new_session_start/sr,
                    "duration": instruction_duration, 
                    "label": os.path.basename(instruction_file_path)[:-4]
                }
                # print(instructions_timings[instruction_order])
        plot_cross_correlation(session_file_path, instruction_file_path, min_max_normalized_cross_corr, peaks_indices)
    
    all_timings = story_timings + instructions_timings
    save_audacity_file(session_file_path, all_timings)

__main__()