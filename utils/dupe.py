# Imports
import argparse
import json
import logging
import math
import os
import string
import sys
import csv
import glob
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import pytorch_lightning as pl
import torch
import torchaudio
import torchaudio.transforms as T
import whisperx
from scipy.signal import butter, correlate, find_peaks, lfilter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import WhisperTimeStampLogitsProcessor, pipeline

from src.config import Config

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pytorch_lightning')
warnings.filterwarnings("ignore", category=UserWarning, module='pyannote')
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
logging.getLogger("torch").setLevel(logging.ERROR)
pl.utilities.rank_zero_only.rank_zero_warn = lambda *args, **kwargs: None

# Assuming config.py is in the project's root directory, similar to experiments.py
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)



















