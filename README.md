# Semantic Sommeliers: Audio Processing and Story Detection System

Semantic Sommeliers is an advanced audio processing system designed to handle complex tasks such as transcription, story detection, and instruction alignment within audio files. Utilizing models like Whisper and WhisperX, the system can transcribe audio, identify stories, and synchronize instructions with the session data effectively.

## Features

- **Audio Transcription:** Leverages OpenAI's Whisper and custom WhisperX models for accurate speech-to-text capabilities.
- **Story Detection:** Identifies and timestamps stories within audio sessions using semantic similarity analysis.
- **Instruction Synchronization:** Aligns instructional audio files with session data, using cross-correlation to find exact timings.
- **Dynamic Configuration:** Allows for varied audio processing settings through external configuration.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or later*
- Poetry Python Package Manager
- FFmpeg for audio processing
- pip
- Conda (optional)

## Installation

To set up the Semantic Sommeliers system on your local machine:

### Using Poetry

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/semantic-sommeliers.git
    cd semantic-sommeliers
    ```

2. **Install Poetry:**

    Follow the instructions on the [Poetry website](https://python-poetry.org/docs/#installation) to install Poetry.

3. **Create and activate the virtual environment:**

    ```bash
    poetry env use python3.10
    ```

4. **Install dependencies:**

    ```bash
    poetry install
    ```

5. **Activate the virtual environment:**

    ```bash
    source $(poetry env info --path)/bin/activate
    ```

6. **Install torch with GPU support:**

    ```bash
    pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
    ```

### Using Conda

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-repository/semantic_sommeliers.git
   cd semantic_sommeliers
    ```

2. **Setup Python Envrionment (using Conda):**

    ```bash
    conda create -n semantic_sommeliers python=3.10
    conda activate semantic_sommeliers
    ```

3. **Install Dependencires:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running Individual Experiments

To run individual experiments with a specific session:

```bash
python experiments.py --session_name [session_filename.wav] --transcript_tool [whisper|whisperx] [optional parameters]
```

Optional parameters and their default values from `config.py` are:

- `--new_sample_rate`: Sample rate for audio processing (default is set in `config.py`)
- `--highcut`: Highcut frequency for filtering (default is set in `config.py`)
- `--lowcut`: Lowcut frequency for filtering (default is set in `config.py`)
- `--normalization`: Enable or disable volume normalization (default is set in `config.py`)
- `--filtering`: Enable or disable filtering (default is set in `config.py`)
- `--seconds_threshold`, `--story_absolute_peak_height`, etc.: Other thresholds and heights as specified in `config.py`

### Running Batch Experiments

To automatically process all session files located in your `data/sessions` directory, run the `run_experiments.py` script. This script reads all `.wav` files in the sessions directory and processes them using the default settings specified in `config.py`:

```bash
python run_experiments.py
```

## Configuration

Modify **'config.py'** to change default settings used by the scripts. These settings include audio processing parameters like sample rate, filter settings, normalization, and detection thresholds. Changes in **'config.py'** will affect both individual and batch processing unless parameters are explicitly overridden in the command line.

## Files and Directories

- **'experiments.py** : Main script for running individual experiments.
- **'run_experiments.py'** : Wrapper script for running experiments in batch mode.
- **'utility/utility.py'** : Contains all utility functions for audio loading, trascription, and other core functionalities

## Contributing

Contributions to improve Semantic Sommeliers are welcome. Please ensure to follow the existing code style and add unit tests for any new or changed functionality.

## License

Distributed under the GNU Lesser General Public License v3.0 (LGPL 3.0). See LICENSE for more information.
