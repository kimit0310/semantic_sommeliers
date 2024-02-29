
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from my_utils.semantic_similarity import SemanticSimilarityCalculator

def main(instructions_json, sessions_json):
    similarity_calculator = SemanticSimilarityCalculator()

    instructions = get_protocol_instructions(instructions_json)
    sessions = get_sessions(sessions_json)

    sessions_scores = []
    for session in sessions:
        session_scores = []
        for text, start, end in session:
            instruction_scores = []
            for instruction in instructions:
                similarity = calculate_similarity(text, instruction, similarity_calculator)
                instruction_scores.append([start, similarity])
                instruction_scores.append([end, similarity])
            session_scores.append(instruction_scores)
        sessions_scores.append(session_scores)

    sessions_starting_times = []
    for session in sessions_scores:
        starting_times = []
        for sentence in session:
            all_similarities = [instruction[1] for instruction in sentence]
            max_value = max(all_similarities)
            max_index = all_similarities.index(max_value)
            if len(starting_times) == 0:
                for _ in range(int(max_index/2)):
                    starting_times.append(None)
                starting_times.append(sentence[max_index])
            elif len(starting_times) > 0 and max_index/2 == len(starting_times):
                starting_times.append(sentence[max_index])
            
            #print(max_value)
            #print(max_index/2)
            #print(sentence[max_index])
        sessions_starting_times.append(starting_times)

    # print(sessions_starting_times)
    os.mkdir('../data/tasks_detected').mkdir(parents=True, exist_ok=True)
    for i, session in enumerate(sessions_starting_times):
        with open(f'../data/tasks_detected/sessions_tasks_start_{i}.txt', 'w') as f:
            for j, task in enumerate(session):
                if task:
                    start = task[0]
                    label = j + 1
                    f.write(f'{start}\t{start}\t{label}\n')



def get_sessions(sessions_json):
    with open(sessions_json, 'r') as f:
        data = json.load(f)
        sessions = []
        for child in data:
            old_start = child['segments'][0]['start']
            old_end = None
            old_text = ""
            old_speaker = child['segments'][0]['speaker']
            session = []
            for segment in child['segments']:
                #print(segment)
                start = segment['start']
                speaker = segment['speaker']
                print(speaker)
                end = segment['end']
                text = segment['text']
                print(text)
                print(old_text)
                if old_speaker != speaker:
                    session.append([old_text, old_start, end])
                    old_start = start
                    old_text = text
                    old_speaker = speaker
                else:
                    old_end = end
                    old_text += text
            sessions.append(session)
    return sessions


def get_protocol_instructions(instructions_json):
    """
    Get the list of instructions from the instructions json file.
    """
    with open(instructions_json, 'r') as f:
        data = json.load(f)
        instructions = []
        for ppt in data:
            text = ""
            for segment in ppt['segments']:
                text += segment['text']
            instructions.append(text)
    return instructions

def calculate_similarity(sentence, instruction, similarity_calculator):
    """
    Calculate semantic similarity between sentences and a list of instructions.
    Returns mean similarities for each sentence.
    """
    sentence_embedding = similarity_calculator.extract_semantic_embeddings(sentence)
    phrase_embedding = similarity_calculator.extract_semantic_embeddings(instruction)
    similarity = similarity_calculator.compute_cosine_similarity(sentence_embedding, phrase_embedding)
    return similarity

def plot_similarity(times, similarities, filename):
    """
    Plot semantic similarity over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(times, similarities, label='Semantic Similarity')
    plt.xlabel('Time')
    plt.ylabel('Similarity')
    plt.title(f'Semantic Similarity Over Time for {filename}')
    plt.legend()
    plt.show()

main("../data/instructions/speech_language_instructions.json", "../data/iktae_test.json")