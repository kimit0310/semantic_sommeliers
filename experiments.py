#!/usr/bin/env python
# coding: utf-8

# Imports
import os
import sys
import warnings
from config import Config
from utility.utility import (
    get_instructions,
    load_audio,
    setup_directories,
    parse_args,
    finalize_results,
    find_stories_in_session,
    load_or_transcribe_audio,
    process_instruction_files,
    process_transcripts,
    tokenize_session_data,
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup environment
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))


def main():
    args = parse_args()
    config = {
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
        audacity_folder,
        transcriptions_folder,
        similarity_folder,
        cross_correlations_folder,
    ) = setup_directories("data", config, args.timestamp)

    # print("Session Name:", args.session_name)
    session_file_path = os.path.join(Config.sessions_folder, args.session_name)

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

    last_instruction_time = max(
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

    # find stories
    story_timings, stories_starts = find_stories_in_session(
        session_word_by_word,
        session_tokens_flat,
        similarity_folder,
        session_file_path,
        full_sr,
        args.story_absolute_peak_height,
    )
    # Adjust story timings by adding the last_instruction_time
    adjusted_story_timings = [
        {**timing, "start": timing["start"] + last_instruction_time}
        for timing in story_timings
        if timing
    ]

    finalize_results(
        audacity_folder, session_file_path, adjusted_story_timings, instructions_timings
    )


if __name__ == "__main__":
    main()

# Try Tiny whisper but maybe not?
