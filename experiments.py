#!/usr/bin/env python
# coding: utf-8

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
from .utility.utility import *

# This is for keeping the console cleaner
warnings.filterwarnings("ignore")

############## PARAMS ##############
data_folder = "data/"
instructions_folder = os.path.join(data_folder, "instructions")
sessions_folder = os.path.join(data_folder, "sessions")
cross_correlations_folder = os.path.join(data_folder, "cross_correlation")
story1 = 'It was the first day of school. It was a tough day for all the kids. One girl had a really hard time because nobody could say her name. Her name was Peggy Babcock. Go ahead. Try and say it three times quickly. “Peggy Babcock Peggy Babcock Peggy Babcock.” Not easy going, right? She was afraid to say hello to any of the other kids on the playground. One boy walked up to her and asked what her name was. She said “When you hear my name it sounds simple but no one can say it. It is Peggy Babcock.” He laughed and said “Your name is tricky but mine is better. It sounds simple but no one can remember it. It is Jonas Norvin Sven Arthur Schwinn Bart Winston Ulysses M.” Peggy laughed and said “Easy. Your name sounds like Joan is nervous when others win. But you win some, you lose some. How do you like my version?" Jonas was so happy that he said “Lets be friends. I will call you PB.” The pair of them stuck so close to each other that everyone at school called them “PB and J.”'
story2 = 'Some time ago, in a place neither near nor far, there lived a king who did not know how to count, not even to zero. Some say this is the reason he would always wish for more — more food, more gold, more land. He simply did not realize how much he already owned. Everyone in his kingdom could do the math and tally bushels of corn, loaves of bread, and urns of gold. But how would they measure the height of his castle or the stretch of his kingdom? You might think “Aaah, ooh, easy — just measure it in meters!” But in those days, the useless unit of measure was based on stains splattered along the king s cloak while drinking shrub juice. The kingdom needed a new way of counting distance. “A kingdom without a proper ruler,” proclaimed the king, “is like riches without measure.” He launched a challenge amid trumpets, drums, flags and cannons. “The person who creates a unit of measure fit for a ruler will be rewarded beyond measure!” A tall order indeed! The first person to come forward was a bulky locksmith with a stiff jaw. He approached the king with an air of secrecy and whispered, “I have the key to measure the kingdom, but only I can wield it.” He then rubbed his beard and pulled the key from his locks of oily hair. The key turned out to be a hair itself! “Judge the reach of my vast kingdom with a hair s width?” laughed the king. “What a poor idea. That would take forever or longer!” The second person eager for the prize was a fidgety boy who knew all numbers (including zero). He produced a curious object from one of his many pockets. It was a complex shape that seemed to change proportions depending on which direction you gazed upon it. The boy said in a measured voice, “This polyhedron has many edges, with each edge of a different length. Only a king could be counted on to use it justly.” He gave the king an awful earful of an explanation that went on and on. The long and the short of it was that the king could make no more use of it than of a puddle of spilled oatmeal. Finally, a little girl with a big idea tugged on the mismeasured cloak of the king. The king sized up the little girl with the big idea and said “I don’t have time for this, and for that matter, I have no concept of space, either.” The girl looked up, then down, then spun around and blurted out: “Aren’t you able to solve the puzzle yourself? Why must you break up your kingdom into tiny pieces when everything around you is Humpty Dumpty together again? Your kingdom IS a unit and you are the ruler.” The king — startled, befuddled, and bemused — found the words wise. He aimed to be satisfied with all around him, big or small or somewhere in between.'
stories = [story1, story2]
############## PARAMS ##############

############## MORE PARAMS TO PLAY WITH ##############
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
############## MORE PARAMS TO PLAY WITH  ##############

def __main__():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Example of reading session_name from CLI parameters.")
    parser.add_argument("session_name", help="Name of the session")
    parser.add_argument("transcript_tool", help="Name of the transcript_tool")
    args = parser.parse_args()

    session_name = args.session_name
    transcript_tool = args.transcript_tool

    audacity_folder = os.path.join(data_folder, f"audacity_{transcript_tool}")
    transcriptions_folder = os.path.join(data_folder, f"transcriptions_{transcript_tool}")
    similarity_folder = os.path.join(data_folder, f"similarity_{transcript_tool}")

    print("Session Name:", session_name)

    session_file_path = os.path.join(sessions_folder, session_name)
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
        if transcript_tool == "whisper":
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
            result = pipe(waveform_session.numpy(), generate_kwargs={"language": "english"})
        else:
            # 1. Transcribe with original whisper (batched)
            whisperx_model = whisperx.load_model("tiny", device, compute_type="int8", language="en") # large-v3

            audio = whisperx.load_audio(session_file_path)
            result = whisperx_model.transcribe(audio, language="en", batch_size=16)

            # 2. Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            result = generate_json(result)
        os.makedirs(transcriptions_folder, exist_ok=True)
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
        story_tokens = [preprocess_and_tokenize(item) for item in story.split(' ')] # story.split('.')[0].split(' ') this would be to look for the first sentence only

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
    save_audacity_file(audacity_folder, session_file_path, all_timings)

__main__()