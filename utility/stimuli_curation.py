from vid_util import list_files, pre_process_audio_from_video
import os
from tqdm import tqdm

def __main__():
    video_format = 'MXF'
    visual_stimuli_folder = '/Users/iktae.kim/Desktop/semantic_sommeliers-dev/data/video_transform'
    local_output_folder = '/Users/iktae.kim/Desktop/semantic_sommeliers-dev/data/sessions'
    audio_format = 'wav'

    if not os.path.exists(local_output_folder):
        os.makedirs(local_output_folder)

    file_list = list_files(visual_stimuli_folder, video_format)

    for file in tqdm(file_list, desc="Processing Files"):
        if not file.lower().endswith(f".{video_format.lower()}"):
            continue # Skip non-video files
        output_filename = os.path.basename(file).replace(f".{video_format}", f".{audio_format}")
        output_path = os.path.join(local_output_folder, output_filename)
        pre_process_audio_from_video(file, output_path)

__main__()