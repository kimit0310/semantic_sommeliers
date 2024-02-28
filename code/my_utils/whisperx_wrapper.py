import whisperx
import gc
import torch
from .utility import get_device
import os

class WhisperxWrapper:
    def __init__(self, model_version="large-v3"):
        self.model_version = model_version
        self.device = get_device()
        self.compute_type = "int8" if self.device == "cpu" else "float16"
        self.faster_whisper = whisperx.load_model(self.model_version, self.device, compute_type=self.compute_type)
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.environ.get('HF_AUTH_TOKEN'), device=self.device)
        
    @staticmethod
    def load_audio(audio_file):
        """
        Loads the audio file.
        """
        return whisperx.load_audio(audio_file)

    def transcribe_audio(self, audio, batch_size=16, language=None):
        """
        Transcribes the audio file using FasterWhisper model.
        """
        result = self.faster_whisper.transcribe(audio, batch_size=batch_size, language=language)
        # print("Transcription (before alignment):", result["segments"])
        return result

    def align_transcript(self, faster_whisper_result, audio, return_char_alignments=False):
        """
        Aligns the transcript using the detected language and Whisper alignment model.
        """
        align_model, align_metadata = whisperx.load_align_model(language_code=faster_whisper_result["language"], device=self.device)
        aligned_result = whisperx.align(faster_whisper_result["segments"], align_model, align_metadata, audio, self.device, return_char_alignments=return_char_alignments)
        # print("Transcription (after alignment):", aligned_result["segments"])
        return aligned_result

    def assign_speaker_labels(self, audio, aligned_result, num_speakers=None, min_speakers=None, max_speakers=None):
        """
        Assigns speaker labels to the aligned transcript segments.
        """
        diarize_segments = self.diarize_model(audio, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        result_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        # print("Diarization segments:", diarize_segments)
        # print("Transcription with speaker labels:", result_with_speakers["segments"])
        return result_with_speakers

    def process_audio_file(self, audio_file, batch_size=16, num_speakers=None, min_speakers=None, max_speakers=None, language=None):
        """
        Main function to process the audio file: transcribes, aligns, and assigns speaker labels.
        """
        audio = self.load_audio(audio_file)
        transcription_result = self.transcribe_audio(audio, batch_size=batch_size, language=language)
        aligned_transcription = self.align_transcript(transcription_result, audio)
        aligned_asigned_transcription = self.assign_speaker_labels(audio, aligned_transcription, num_speakers, min_speakers, max_speakers)

        # Cleanup to free up resources if needed
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return aligned_asigned_transcription
