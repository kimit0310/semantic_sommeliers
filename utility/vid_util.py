import ffmpeg
import torchaudio
import torch
import os
import glob
import re
import pyloudnorm as pyln
import numpy as np
import json
import tempfile

def extract_audio_from_video(video_path):
    """
    Extracts audio from video with no compression and returns audio bytes.
    """
    stream = ffmpeg.input(video_path)
    mixed_audio = ffmpeg.filter([stream], 'amix', inputs=2, duration='longest')
    audio = mixed_audio.output('pipe:', format='wav', acodec='pcm_s16le')
    try:
        out, _ = ffmpeg.run(audio, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print("ffmpeg error:", e.stderr.decode('utf-8'))
        raise e
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

def load_audio_from_bytes(audio_bytes, format):
    """
    Loads audio from bytes by first writing to a temporary file, then loading it.
    """
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio_file:
        # Write the audio bytes to a temporary file
        temp_audio_file.write(audio_bytes)
        temp_audio_file.seek(0)  # Go back to the start of the file
        
        # Load the audio from the temporary file
        waveform, sample_rate = torchaudio.load(temp_audio_file.name, format=format)
        
    return waveform, sample_rate

def normalize_loudness(audio, rate, target_loudness=-23):
    """
    Normalizes the loudness of the audio to target_loudness.
    """
    meter = pyln.Meter(rate)  # create a BS.1770 meter
    current_loudness = meter.integrated_loudness(np.array(audio.squeeze()))
    # Normalize the loudness of the audio to the target loudness level
    loudness_normalized_audio = pyln.normalize.loudness(np.array(audio.squeeze()), current_loudness, target_loudness)
    return torch.tensor(loudness_normalized_audio).unsqueeze(0)

def natural_sort_key(s):
    """
    Obtain a tuple that represents the natural order sort key of the input string.
    
    Args:
        s (str): The string to parse and turn into a sort key.
    
    Returns:
        tuple: A tuple of mixed integer and string parts for sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def list_files(folder_path, format):
    path_pattern = os.path.join(folder_path, '**', '*.' + format)
    files = glob.glob(path_pattern, recursive=True)
    files.sort(key=natural_sort_key)
    return files  # Use natural sorting to maintain order

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def save_json(file, results):
    with open(file, 'w') as f:
        json.dump(results, f)

def process_and_concatenate_videos(video_files, output_path, target_sample_rate=16000):
    concatenated_waveform = []
    for video_file in video_files:
        audio_bytes = extract_audio_from_video(video_file)
        waveform, sample_rate = load_audio_from_bytes(audio_bytes, 'wav')
        waveform_mono = convert_stereo_to_mono(waveform)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform_mono)
        concatenated_waveform.append(waveform)

    concatenated_waveform = torch.cat(concatenated_waveform, dim=1)
    save_audio(concatenated_waveform, target_sample_rate, output_path)
