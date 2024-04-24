from vid_util import list_files, process_and_concatenate_videos
import os
from tqdm import tqdm

def __main__():
    video_format = 'MXF'
    visual_stimuli_folder = '/Users/iktae.kim/semantic_sommeliers/data/video_transform'
    local_output_folder = '/Users/iktae.kim/semantic_sommeliers/data/sessions_2'
    audio_format = 'wav'
    target_sample_rate = 16000

    if not os.path.exists(local_output_folder):
        os.makedirs(local_output_folder)

    file_list = list_files(visual_stimuli_folder, video_format) # list_file innately has nat sort, don't think we need another sort in ln 25, check if both outputs are the same.
    grouped_files = {}

    for file in file_list:
        participant_id = os.path.basename(file)[:7] # Using first 7 as "Participant ID"
        if participant_id not in grouped_files:
            grouped_files[participant_id] = []
        grouped_files[participant_id].append(file)

    for participant_id, files in tqdm(grouped_files.items(), desc="Processing Participant Files"):
        files.sort()  # Ensure the files are in the correct order if not already
        output_filename = f'{participant_id}_concatenated.{audio_format}' # Switch to _speech_language if needed
        output_path = os.path.join(local_output_folder, output_filename)
        process_and_concatenate_videos(files, output_path, target_sample_rate)

__main__()
