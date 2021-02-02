import backbone.support.configurations_variables as confv
import os


def check_dir(dir):
    # Store Hierarchy
    dir_pth = ""
    if dir == confv.base_store:
        if not os.path.isdir(dir):
            os.makedirs(dir)
        dir_pth = dir
    elif dir in {confv.saved_dataframes, confv.saved_features, confv.saved_models, confv.saved_modelconfigs, confv.saved_training_metrics_logs}:
        dir_pth = check_dir(confv.base_store)
        dir_pth = os.path.join(dir_pth, dir)
        if not os.path.isdir(dir_pth):
            os.makedirs(dir_pth)
    elif dir == confv.clean_audio:
        dir_pth = check_dir(confv.base_store)
        dir_pth = os.path.join(dir_pth, dir)
        if not os.path.isdir(dir_pth):
            os.makedirs(dir_pth)
    elif dir in {confv.dataset_ravdess_male, confv.dataset_ravdess_female, confv.dataset_emodb_male, confv.dataset_emodb_female, confv.dataset_cremad_male, confv.dataset_cremad_female, confv.dataset_shemo_male, confv.dataset_shemo_female}:
        dir_pth = check_dir(confv.clean_audio)
        dir_pth = os.path.join(dir_pth, dir)
        if not os.path.isdir(dir_pth):
            os.makedirs(dir_pth)

    return dir_pth


def check_file_existence(file_path):
    if os.path.isfile(file_path):
        return True


def check_dir_inside_saved_features_and_modelconfigs_and_models(parent, database, gender):
    dir_pth = check_dir(parent)
    dir_pth = os.path.join(dir_pth, database)
    if not os.path.isdir(dir_pth):
        os.makedirs(dir_pth)
    dir_pth = os.path.join(dir_pth, gender)
    if not os.path.isdir(dir_pth):
        os.makedirs(dir_pth)
