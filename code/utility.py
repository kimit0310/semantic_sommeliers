import ffmpeg
import torchaudio
import torch
import os
import glob
import logging
import io
import tempfile

def extract_audio_from_video(video_path):
    """
    Extracts audio from video with no compression and returns audio bytes.
    """
    stream = ffmpeg.input(video_path)
    audio = stream.audio.output('pipe:', format='wav', acodec='pcm_s16le')
    out, _ = ffmpeg.run(audio, capture_stdout=True, capture_stderr=True)
    return out

def convert_stereo_to_mono(waveform):
    """
    Converts stereo audio waveform to mono by averaging the channels.
    """
    if waveform.shape[0] > 1:  # Check if the audio is stereo
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def resample_audio(waveform, orig_sample_rate, new_sample_rate=16000):
    """
    Resamples the audio waveform to a new sample rate.
    """
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=new_sample_rate)
    waveform_resampled = resampler(waveform)
    return waveform_resampled, new_sample_rate

def save_audio(waveform, sample_rate, output_path):
    """
    Saves the audio waveform to a WAV file with 16 bits per sample.
    """
    torchaudio.save(output_path, waveform, sample_rate, bits_per_sample=16)

def process_audio_from_video(video_path, output_path):
    """
    Extracts audio from a video file, converts it to mono and 16kHz sample rate,
    and saves it as a 16-bit WAV file.
    """
    audio_bytes = extract_audio_from_video(video_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_file_path = temp_audio_file.name

    waveform, sample_rate = torchaudio.load(temp_audio_file_path, format='wav')
    waveform_mono = convert_stereo_to_mono(waveform)
    waveform_resampled, new_sample_rate = resample_audio(waveform_mono, sample_rate)
    save_audio(waveform_resampled, new_sample_rate, output_path)

def list_files(folder_path, format):
    # Build the path pattern to match all .mp4 files in the folder and subfolders
    path_pattern = os.path.join(folder_path, '**', '*.{}'.format(format))
    
    # Use glob.glob with recursive=True to find all files matching the pattern in subfolders as well
    mp4_files = glob.glob(path_pattern, recursive=True)
    
    return mp4_files

def __main__():
    file_list = list_files('data/speech_language_instructions', 'mp4')
    for file in file_list:
        output_path = file.replace('.mp4', '.wav').replace('speech_language_instructions', 'speech_language_instructions_wav')
        process_audio_from_video(file, output_path)

__main__()