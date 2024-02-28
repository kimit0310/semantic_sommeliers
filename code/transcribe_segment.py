from my_utils.utility import *
import json
from my_utils.whisperx_wrapper import WhisperxWrapper

def __main__():
    audio_stimuli_folder = '../data/session'
    audio_format = 'wav'
    whisper_model_version = 'large-v3'
    task_type = 'linguistic_task'
    output_file = '../data/iktae_test.json'

    prepare_env()
    whisperxWrapper = WhisperxWrapper(model_version=whisper_model_version)
    audio_files = list_files(audio_stimuli_folder, audio_format)
    results = []
    for audio_order, audio_file in enumerate(sorted(audio_files)):
        transcription_result = whisperxWrapper.process_audio_file(audio_file, 
                                                                  batch_size=16, 
                                                                  num_speakers=2, 
                                                                  min_speakers=0, 
                                                                  max_speakers=2,
                                                                  language="en")
        results.append({
            "order": audio_order,
            "file_name": audio_file,
            "task_type": task_type,
            "segments": transcription_result["segments"]
        })

    save_json(output_file, results)

__main__()