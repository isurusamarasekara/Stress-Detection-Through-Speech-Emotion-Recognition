from speech_analysis_raspi.support import configurations_variables as confv
import os
import sys
from speech_analysis_raspi.support import calculations as calc


def check_dir(directory):
    # Store Hierarchy
    dir_pth = ""
    if directory == confv.preprediction_base_audio_store:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        dir_pth = directory
    elif directory in {confv.prerecorded_upload_dir, confv.prerecorded_time_specified_dir, confv.prerecorded_time_specified_noisy_clips_dir}:
        dir_pth = check_dir(directory=confv.preprediction_base_audio_store)
        dir_pth = os.path.join(dir_pth, directory)
        if not os.path.isdir(dir_pth):
            os.makedirs(dir_pth)

    return dir_pth


def clean_relevant_directory(directory):
    dir_path = ""
    if directory == confv.prerecorded_upload_dir:
        dir_path = confv.prerecorded_upload_dir_pth
    elif directory == confv.prerecorded_time_specified_dir:
        dir_path = confv.prerecorded_time_specified_dir_pth

    for fl_name in os.listdir(dir_path):
        fl_pth = os.path.join(dir_path, fl_name)
        os.remove(fl_pth)


def is_directory_empty(directory):
    dir_path = ""
    if directory == confv.prerecorded_upload_dir:
        dir_path = confv.prerecorded_upload_dir_pth
    elif directory == confv.prerecorded_time_specified_dir:
        dir_path = confv.prerecorded_time_specified_dir_pth
    elif directory == confv.prerecorded_time_specified_noisy_clips_dir:
        dir_path = confv.prerecorded_time_specified_noisy_clips_dir_pth

    isEmpty = True
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        if not os.listdir(dir_path):
            isEmpty = True
            # print("Directory is empty")
        else:
            isEmpty = False
            # print("Directory is not empty")
    else:
        print("A directory does not exists by the specified name")
        sys.exit()

    return isEmpty


def get_audio_file_path_from_directory(directory):
    '''
    dir_path = ""
    if directory == confv.prerecorded_upload_dir:
        dir_path = confv.prerecorded_upload_dir_pth
    elif directory == confv.prerecorded_realtime_dir:
        dir_path = confv.prerecorded_realtime_dir_pth

    # If cleaning is properly executed, there will not be multiple files. But just in case below is coded.
    aud_fl_list = os.listdir(dir_path)
    aud_fl_list.sort()
    # print(aud_fl_list)
    aud_fl_name = aud_fl_list[-1]

    aud_fl_pth = os.path.join(dir_path, aud_fl_name)
    '''

    aud_fl_pth = ''
    if directory == confv.prerecorded_upload_dir:
        aud_fl_pth = confv.prerecorded_upload_file_aud_fl_pth
    elif directory == confv.prerecorded_time_specified_dir:
        aud_fl_pth = confv.prerecorded_time_specified_file_aud_fl_pth

    return aud_fl_pth


def check_for_minimum_length_compliance(signal, threshold, application_type):
    signal = calc.apply_envelope(signal=signal, threshold=threshold, application_type=application_type)

    # For below to be the condition, step_size should be same as in the build features in model training
    # step_size is bound to build features in model training.
    isComplied = False
    if (application_type == confv.app_type_prerecorded_upload or
        application_type == confv.app_type_prerecorded_time_specified or
        application_type == confv.app_type_prerecorded_time_specified_noise or
        application_type == confv.app_type_real_time) and signal.shape[0] > confv.step_size:
        # The models have been trained on confv.step_size length feature sets.
        # Hence, what ever the application_type, the compliance condition must be based on the confv.step_size that
        # represents the same in backbone.build_features
        isComplied = True
        # print("The audio file is in compliance with the minimum audio length requirement")

    else:
        # print("The audio file is NOT in compliance with the minimum audio length requirement. The length of the audio file is insufficient")
        pass

    return isComplied, signal


def check_file_existence(file_path):
    doesExist = False
    if os.path.exists(file_path):
        # print("Something exists in the path", file_path)

        if os.path.isfile(file_path):
            doesExist = True
            # print("A file exists by the path", file_path)
        elif os.path.isdir(file_path):
            doesExist = True
            # print("A directory exists by the path", file_path)

    else:
        print("A file nor directory exist by the path", file_path)

    return doesExist
