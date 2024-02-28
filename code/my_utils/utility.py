import ffmpeg
import torchaudio
import torch
import os
import glob
import io
from dotenv import load_dotenv
import pyloudnorm as pyln
import numpy as np
import json

def prepare_env():
    load_dotenv()

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, waveform, sample_rate, bits_per_sample=16)

def pre_process_audio_from_video(video_path, output_path):
    """
    Extracts audio from a video file, converts it to mono and 16kHz sample rate,
    and saves it as a 16-bit WAV file.
    """
    audio_bytes = extract_audio_from_video(video_path)
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes), format='wav')
    waveform_mono = convert_stereo_to_mono(waveform)
    waveform_resampled, new_sample_rate = resample_audio(waveform_mono, sample_rate)
    normalized_waveform = normalize_loudness(waveform_resampled, new_sample_rate, target_loudness=-23)
    save_audio(normalized_waveform, new_sample_rate, output_path)

def normalize_loudness(audio, rate, target_loudness=-23):
    """
    Normalizes the loudness of the audio to target_loudness.
    """
    meter = pyln.Meter(rate)  # create a BS.1770 meter
    current_loudness = meter.integrated_loudness(np.array(audio.squeeze()))
    # Normalize the loudness of the audio to the target loudness level
    loudness_normalized_audio = pyln.normalize.loudness(np.array(audio.squeeze()), current_loudness, target_loudness)
    return torch.tensor(loudness_normalized_audio).unsqueeze(0)

def list_files(folder_path, format):
    # Build the path pattern to match all .mp4 files in the folder and subfolders
    path_pattern = os.path.join(folder_path, '**', '*.{}'.format(format))
    
    # Use glob.glob with recursive=True to find all files matching the pattern in subfolders as well
    mp4_files = glob.glob(path_pattern, recursive=True)
    
    return mp4_files

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def save_json(file, results):
    with open(file, 'w') as f:
        json.dump(results, f)