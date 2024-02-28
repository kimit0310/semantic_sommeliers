
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

    print(instructions)
    print(sessions)
    input("hhh")

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

    print(sessions_scores)
    input("hhh")


    for session in sessions_scores:
        for sentence in session:
            all_similarities = [instruction[1] for instruction in sentence]
            max_value = max(all_similarities)
            max_index = all_similarities.index(max_value)
            print(max_value)
            print(max_index/2)
            print(sentence[max_index])

    input("hhh")


    sessions_scores = []
    for session in sessions_json:
        session_scores = []
        for instruction in instructions:
            instructions_scores = {}
            for utterance, start, end in get_utterances(session):
                instruction_scores = []
                similarity = calculate_similarity(utterance, instruction, similarity_calculator)
                instruction_scores.append([start, similarity])
                instruction_scores.append([end, similarity])
        session_scores.append(instruction_scores)

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