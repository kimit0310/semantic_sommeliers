from my_utils.utility import list_files, pre_process_audio_from_video
def __main__():
    video_format = 'MXF'
    visual_stimuli_folder = '../data/sessions'
    audio_format = 'wav'

    file_list = list_files(visual_stimuli_folder, video_format)
    for file in file_list:
        output_path = file.replace(f".{video_format}", f".{audio_format}").replace(visual_stimuli_folder, f"{visual_stimuli_folder}_{audio_format}")
        pre_process_audio_from_video(file, output_path)

__main__()