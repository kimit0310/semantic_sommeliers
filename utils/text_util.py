"""
This module contains utilities for processing and transcribing audio files, as well as extracting and analyzing
semantic information from transcriptions. It includes functions for loading or transcribing audio, processing 
transcripts, tokenizing text, extracting semantic embeddings, computing cosine similarity, and identifying stories 
within session data based on semantic similarity.

Functions:
    - load_or_transcribe_audio: Load or transcribe audio based on the necessity of transcription.
    - transcribe_with_whisper: Transcribe a given waveform using Whisper.
    - transcribe_with_whisperx: Transcribe a given waveform using WhisperX and format the output.
    - process_transcripts: Process raw transcript data into a structured format.
    - preprocess_and_tokenize: Preprocess and tokenize text by removing punctuation and converting to lowercase.
    - extract_semantic_embeddings: Extract semantic embeddings from a sentence using a given model.
    - compute_cosine_similarity: Compute the cosine similarity between two sets of embeddings.
    - tokenize_session_data: Tokenize the text from the session transcripts.
    - find_story_in_session: Find a story within the session using semantic similarity.
    - find_stories_in_session: Find and time stories within the session data without adjusting the waveform session.
"""

import json
import os
import string
from typing import List, Optional, Dict, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import whisperx # type: ignore
from scipy.signal import find_peaks # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from transformers import WhisperTimeStampLogitsProcessor, pipeline # type: ignore

from src.config import Config
from utils.general_util import generate_json, min_max_normalization

def load_or_transcribe_audio(
    session_file_path: str,
    waveform: torch.Tensor,
    sample_rate: int,
    transcript_tool: str,
    transcriptions_folder: str,
    last_instruction_time: float,
) -> dict:
    """
    Load or transcribe audio based on the necessity of transcription.

    Args:
        session_file_path (str): Path to the session file.
        waveform (torch.Tensor): Waveform of the audio.
        sample_rate (int): Sample rate of the audio.
        transcript_tool (str): Tool used for transcription ('whisper' or 'whisperx').
        transcriptions_folder (str): Folder to save transcriptions.
        last_instruction_time (float): Time of the last instruction in the session.

    Returns:
        dict: Transcription result.
    """
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


def transcribe_with_whisper(waveform: torch.Tensor, device: str) -> dict:
    """
    Transcribe a given waveform using Whisper.

    Args:
        waveform (torch.Tensor): Waveform of the audio.
        device (str): Device to use for transcription ('cuda' or 'cpu').

    Returns:
        dict: Transcription result.
    """
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    processor = WhisperTimeStampLogitsProcessor.from_pretrained(model_id) # type: ignore
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


def transcribe_with_whisperx(waveform: Union[torch.Tensor, np.ndarray], device: str) -> dict:
    """
    Transcribe a given waveform using WhisperX and format the output.

    Args:
        waveform (Union[torch.Tensor, np.ndarray]): Waveform of the audio.
        device (str): Device to use for transcription ('cuda' or 'cpu').

    Returns:
        dict: Transcription result.
    """
    whisperx_model = whisperx.load_model(
        "large", device, compute_type="int8", language="en"
    )

    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    if waveform.size == 0:
        print("Warning: Received empty waveform for transcription.")
        return {} 

    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32) / np.max(np.abs(waveform))

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


def process_transcripts(result: dict) -> List[Dict[str, Union[str, float, None]]]:
    """
    Process raw transcript data into a structured format.

    Args:
        result (dict): Raw transcription result.

    Returns:
        List[Dict[str, Union[str, float, None]]]: Processed transcript data.
    """
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

def preprocess_and_tokenize(text: str) -> List[str]:
    """
    Preprocess and tokenize text by removing punctuation and converting to lowercase.

    Args:
        text (str): Input text.

    Returns:
        List[str]: List of tokens.
    """
    clean_text = "".join(
        char.lower() if char.isalnum() or char.isspace() else " " for char in text
    )
    tokens = clean_text.split()
    return tokens


def extract_semantic_embeddings(model: SentenceTransformer, sentence: str) -> np.ndarray:
    """
    Extract semantic embeddings from a sentence using a given model.

    Args:
        model (SentenceTransformer): Sentence transformer model.
        sentence (str): Input sentence.

    Returns:
        np.ndarray: Semantic embeddings.
    """
    return model.encode(sentence)


def compute_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two sets of embeddings.

    Args:
        embeddings1 (np.ndarray): First set of embeddings.
        embeddings2 (np.ndarray): Second set of embeddings.

    Returns:
        float: Cosine similarity score.
    """
    cosine_sim = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1))
    return np.mean(cosine_sim)

def tokenize_session_data(session_word_by_word: List[Dict[str, Union[str, float, None]]]) -> List[str]:
    """
    Tokenize the text from the session transcripts.

    Args:
        session_word_by_word (List[Dict[str, Union[str, float, None]]]): Session transcript data.

    Returns:
        List[str]: List of tokens.
    """
    session_tokens = [
        preprocess_and_tokenize(item["text"]) for item in session_word_by_word if isinstance(item["text"], str)
    ]
    session_tokens_flat = [item for sublist in session_tokens for item in sublist]
    return session_tokens_flat


def find_story_in_session(
    session_transcript: List[Dict[str, Union[str, float, None]]],
    session_tokens: List[str],
    story_tokens: List[str],
    story_absolute_peak_height: float = 0.65,
    file_path: Optional[str] = None,
) -> Optional[float]:
    """
    Find a story within the session using semantic similarity.

    Args:
        session_transcript (List[Dict[str, Union[str, float, None]]]): Session transcript data.
        session_tokens (List[str]): Session tokens.
        story_tokens (List[str]): Story tokens.
        story_absolute_peak_height (float, optional): Absolute peak height threshold.
        file_path (Optional[str], optional): Path to save the plot.

    Returns:
        Optional[float]: Start time of the found story, if any.
    """
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    times = [
        session_transcript[i]["start"]
        for i in range(len(session_tokens) - len(story_tokens) + 1)
        if i < len(session_transcript) and isinstance(session_transcript[i]["start"], float)
    ]
    times = [t for t in times if isinstance(t, float)]
    similarities = []
    for i, _ in enumerate(times):
        embeddings1 = extract_semantic_embeddings(model, " ".join(session_tokens[i : i + len(story_tokens)]))
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
        plt.scatter(float(times[i]), normalized_similarities[i], color="red", zorder=5) # type: ignore
        if file_path:
            plt.savefig(file_path)
        return float(times[i]) # type: ignore

    if file_path:
        plt.savefig(file_path)
    return None


def find_stories_in_session(
    session_word_by_word: List[Dict[str, Union[str, float, None]]],
    session_tokens_flat: List[str],
    similarity_folder: str,
    session_file_path: str,
    sr: int,
    story_absolute_peak_height: float,
) -> Tuple[List[Optional[Dict[str, Union[int, float, str]]]], List[float]]:
    """
    Find and time stories within the session data without adjusting the waveform session.

    Args:
        session_word_by_word (List[Dict[str, Union[str, float, None]]]): Session transcript data.
        session_tokens_flat (List[str]): Session tokens.
        similarity_folder (str): Folder to save similarity plots.
        session_file_path (str): Path to the session file.
        sr (int): Sample rate.
        story_absolute_peak_height (float): Absolute peak height threshold.

    Returns:
        Tuple[List[Optional[Dict[str, Union[int, float, str]]]], List[float]]: List of story timings and story start times.
    """
    story_timings: List[Optional[Dict[str, Union[int, float, str]]]] = [None] * len(Config.stories)
    stories_starts: List[float] = []
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