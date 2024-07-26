#!/usr/bin/env python
# coding: utf-8
"""
This script executes the audio and text processing pipeline for a session data. It handles argument parsing,
directory setup, audio loading, instruction processing, audio transcription, session data tokenization,
story detection within the session, and final result finalization.

Functions:
    - main: Main function to execute the audio and text processing pipeline.
"""

import os
import sys
from typing import Dict, Union

from src.config import Config
from utils.general_util import (
    get_instructions,
    setup_directories,
    parse_args,
    finalize_results,
    summarize_results
)
from utils.audio_util import (
    load_audio,
    process_instruction_files,
)
from utils.text_util import (
    find_stories_in_session,
    load_or_transcribe_audio,
    process_transcripts,
    tokenize_session_data
)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

def main() -> None:
    """
    Main function to execute the audio and text processing pipeline.
    
    It parses arguments, sets up directories, loads audio data, processes instructions,
    transcribes audio, tokenizes session data, finds stories in the session, and finalizes results.
    """
    args = parse_args()
    config: Dict[str, Union[int, float, bool]] = {
        "new_sample_rate": args.new_sample_rate,
        "highcut": args.highcut,
        "lowcut": args.lowcut,
        "normalization": args.normalization,
        "filtering": args.filtering,
        "seconds_threshold": args.seconds_threshold,
        "story_absolute_peak_height": args.story_absolute_peak_height,
        "long_instructions_peak_height": args.long_instructions_peak_height,
        "word_instructions_peak_height": args.word_instructions_peak_height,
        "non_word_instructions_peak_height": args.non_word_instructions_peak_height,
        "long_instructions_absolute_peak_height": args.long_instructions_absolute_peak_height,
        "word_instructions_absolute_peak_height": args.word_instructions_absolute_peak_height,
        "non_word_instructions_absolute_peak_height": args.non_word_instructions_absolute_peak_height,
    }

    (
        labels_folder,
        transcriptions_folder,
        similarity_folder,
        cross_correlations_folder,
    ) = setup_directories("/data3/mobi/hbn_video_qa/qa_data", config, args.timestamp)

    session_file_path: str = os.path.join(Config.sessions_folder, args.session_name)

    full_waveform, full_sr, _ = load_audio(
        session_file_path,
        new_sample_rate=args.new_sample_rate,
        filtering=args.filtering,
        lowcut=args.lowcut,
        highcut=args.highcut,
        normalization=args.normalization,
    )

    instruction_file_paths = get_instructions(Config.instructions_folder)
    instructions_timings = process_instruction_files(
        instruction_file_paths,
        full_waveform,
        full_sr,
        session_file_path,
        cross_correlations_folder,
    )

    last_instruction_time: float = max(
        (timing["start"] + timing["duration"])
        for timing in instructions_timings
        if timing
    )

    transcription_result = load_or_transcribe_audio(
        session_file_path,
        full_waveform,
        full_sr,
        args.transcript_tool,
        transcriptions_folder,
        last_instruction_time,
    )

    session_word_by_word = process_transcripts(transcription_result)
    session_tokens_flat = tokenize_session_data(session_word_by_word)

    story_timings, stories_starts = find_stories_in_session(
        session_word_by_word,
        session_tokens_flat,
        similarity_folder,
        session_file_path,
        full_sr,
        args.story_absolute_peak_height,
    )
    adjusted_story_timings = [
        {**timing, "start": float(timing["start"]) + last_instruction_time}
        for timing in story_timings
        if timing
    ]

    finalize_results(
        labels_folder, session_file_path, adjusted_story_timings, [timing for timing in instructions_timings if timing]
    )
    
    summarize_results(labels_folder)

if __name__ == "__main__":
    main()
