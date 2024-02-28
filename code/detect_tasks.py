
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from my_utils.semantic_similarity import SemanticSimilarityCalculator

def main(task_folder, instructions_json, sessions_json):
    # for each task (json file in the task folder)
    # get the list of instructions
    # for each video_transcript (json file in the video folder),
    # calculate the semantic similarity between the sentences in the video_transcript and the instructions 
    # plot the similarity (y) over time (x)

    similarity_calculator = SemanticSimilarityCalculator()

    instructions = get_protocol_instructions(instructions_json)
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