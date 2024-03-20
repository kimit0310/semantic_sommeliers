#!/usr/bin/env python
# coding: utf-8

import torch
import torchaudio
import os
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import torch.nn.functional as F
from IPython.display import Audio
import pyloudnorm as pyln
from torchaudio.functional import highpass_biquad
from scipy.signal import butter, lfilter, freqz
import numpy as np
from tqdm import tqdm
import warnings
from torchaudio import transforms

warnings.filterwarnings("ignore")

data_folder = "data/"
instructions_folder = os.path.join(data_folder, "instructions")
sessions_folder = os.path.join(data_folder, "sessions")
cross_correlations_folder = os.path.join(data_folder, "cross_correlation")
audacity_folder = os.path.join(data_folder, "audacity")

normalization = True
filtering = True
new_sample_rate = 8000
lowcut = 300.0
highcut = 3400.0
correlation_threshold = 25.0
seconds_threshold = 5.0

os.makedirs(cross_correlations_folder, exist_ok=True)
os.makedirs(audacity_folder, exist_ok=True)

def load_audio(path, new_sample_rate=None):
    waveform, sample_rate = torchaudio.load(path)
    if new_sample_rate:
        resampler = T.Resample(sample_rate, new_sample_rate, dtype=waveform.dtype)
        waveform = resampler(waveform)
        return waveform, new_sample_rate
    return waveform, sample_rate


def normalize_volume(data, rate, new_loudness=-23):   
    # measure the loudness first 
    meter = pyln.Meter(rate) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)
    
    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -12.0)
    return loudness_normalized_audio
    


def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, _ = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)


def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq  # Normalize lowcut
    high = highcut / nyq  # Normalize highcut
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def normalize_cross_correlation(cross_corr, signal1, signal2):
    """
    Normalize the cross-correlation values to be between 0 and 1.

    Parameters:
    - cross_corr: The computed cross-correlation values.
    - signal1: The first signal (array).
    - signal2: The second signal (array).

    Returns:
    - Normalized cross-correlation array.
    """
    # Compute the energy (sum of squares) of each signal
    energy_signal1 = np.sum(signal1 ** 2)
    energy_signal2 = np.sum(signal2 ** 2)

    # Normalize the cross-correlation values
    normalization_factor = np.sqrt(energy_signal1 * energy_signal2)
    normalized_cross_corr = cross_corr / normalization_factor

    return normalized_cross_corr


def normalize_array_to_0_1(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr

def standardize_array(arr):
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    standardized_arr = (arr - arr_mean) / arr_std
    return standardized_arr

def min_max_normalization(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr

def save_cross_correlation(cross_correlation, file_path):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(cross_correlation, label='Normalized Cross-Correlation')
    plt.xlabel('Lag')
    plt.ylabel('Normalized Cross-Correlation')
    plt.title('Normalized Cross-Correlation between Two Signals')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(file_path)


def cross_correlation_fft(signal1, signal2):
    # Pad signals
    n = len(signal1) + len(signal2) - 1
    padded_signal1 = np.pad(signal1, (0, n - len(signal1)))
    padded_signal2 = np.pad(signal2, (0, n - len(signal2)))
    
    # FFT of both signals
    fft_signal1 = np.fft.fft(padded_signal1)
    fft_signal2 = np.fft.fft(padded_signal2)
    
    # Pointwise multiply and inverse FFT
    result = np.fft.ifft(fft_signal1 * np.conj(fft_signal2))
    
    # Return the real part (to handle numerical imprecisions)
    return np.real(result)


instruction_file_paths = [os.path.join(instructions_folder, instruction_filename) for instruction_filename in os.listdir(instructions_folder) if instruction_filename.endswith(".wav")]
session_file_paths = [os.path.join(sessions_folder, session_filename) for session_filename in os.listdir(sessions_folder) if session_filename.endswith(".wav")]

transform = transforms.Vad(sample_rate=new_sample_rate)


sessions_peaks = []
for session_file_path in tqdm(sorted(session_file_paths), desc='Sessions folders', leave=False):
    peaks = []
    
    # print(sessions_folder)
    waveform_session, sr = load_audio(session_file_path, new_sample_rate=new_sample_rate)
    if filtering:
        waveform_session = torch.tensor(bandpass_filter(waveform_session, lowcut=lowcut, highcut=highcut, fs=sr))
    if normalization:
        waveform_session = torch.tensor(normalize_volume(waveform_session[0].numpy(), sr).reshape(1, -1))
    waveform_session = waveform_session.squeeze()

    for instruction_file_path in tqdm(sorted(instruction_file_paths), desc='Instruction files'):
        # print(instruction_file_path)
        waveform_instruction, sr = load_audio(instruction_file_path, new_sample_rate=new_sample_rate)
        instruction_duration = waveform_instruction.shape[1]/new_sample_rate
        # print(f"Duration: {instruction_duration}")
        if filtering:
            waveform_instruction = torch.tensor(bandpass_filter(waveform_instruction, lowcut=lowcut, highcut=highcut, fs=sr))
        if normalization:
            waveform_instruction = torch.tensor(normalize_volume(waveform_instruction[0].numpy(), sr).reshape(1, -1))
        waveform_instruction = waveform_instruction.squeeze()

        # optimization to speed up the process
        if waveform_instruction.shape[0]/sr > seconds_threshold:
            # print("Instruction too long")
            waveform_instruction = waveform_instruction[:int(seconds_threshold*sr)]

        waveform_instruction = torch.flip(waveform_instruction, dims=[0])
        waveform_instruction = transform(waveform_instruction)
        waveform_instruction = torch.flip(waveform_instruction, dims=[0])

        if len(peaks) > 0:
            my_new_session_start = int(peaks[-1][0] + peaks[-1][1])*sr
            my_waveform_session = waveform_session[my_new_session_start:]
        else:
            my_new_session_start = 0
            my_waveform_session = waveform_session
        # print("Start computing the correlation...")
        cross_correlation = np.correlate(my_waveform_session, waveform_instruction, mode='full')
        #max_corr_index = np.argmax(cross_correlation)
        #max_corr_value = cross_correlation[max_corr_index]
        # normalized_cross_corr = normalize_cross_correlation(cross_correlation, my_waveform_session.numpy(), waveform_instruction.numpy())
        #max_corr_index = np.argmax(standardized_normalized_cross_corr)
        #max_corr_value = standardized_normalized_cross_corr[max_corr_index]
        #normalized_standardized_cross_corr = min_max_normalization(cross_correlation)
        standardized_normalized_cross_corr = standardize_array(cross_correlation)

        max_corr_index = np.argmax(standardized_normalized_cross_corr)
        max_corr_value = standardized_normalized_cross_corr[max_corr_index]
        if max_corr_value > correlation_threshold:
            # print(f"Maximum standardized correlation at index: {max_corr_index}, value: {max_corr_value}")
            shift = (max_corr_index - (len(waveform_instruction) - 1))/ sr
            print(f"Shift: {shift + my_new_session_start/sr} seconds")
            peaks.append([shift + my_new_session_start/sr, instruction_duration, os.path.basename(instruction_file_path)[:-4], max_corr_value])
        
        cross_correlations_session_folder = os.path.join(cross_correlations_folder, os.path.basename(session_file_path)[:-4])
        os.makedirs(cross_correlations_session_folder, exist_ok=True)
        # save_cross_correlation(standardized_normalized_cross_corr, os.path.join(cross_correlations_folder, f"{os.path.basename(instruction_file_path)[:-4]}_{os.path.basename(session_file_path)[:-4]}.png"))        
        save_cross_correlation(standardized_normalized_cross_corr, os.path.join(cross_correlations_session_folder, f"{os.path.basename(instruction_file_path)[:-4]}_{os.path.basename(session_file_path)[:-4]}.svg"))
    sessions_peaks.append(peaks)

for i, session_peaks in enumerate(sessions_peaks):
    with open(f'{audacity_folder}/{os.path.basename(sorted(session_file_paths)[i])[:-4]}.txt', 'w', encoding='utf-8') as file:
        for item in session_peaks:
            file.write(f"{item[0]}\t{item[0]+item[1]}\t{item[2]}\n")  # __{item[3]}

