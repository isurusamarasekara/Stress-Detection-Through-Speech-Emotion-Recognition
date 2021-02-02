import os

# Support Directories
base_store = '../base_store'
saved_dataframes = 'saved_dataframes'
saved_features = 'saved_features'
saved_modelconfigs = 'saved_modelconfigs'
saved_models = 'saved_models'
clean_audio = 'clean_audio'
saved_training_metrics_logs = 'saved_training_metrics_logs'
log_dir = 'logs'

# Genders
gender_male = 'male'
gender_female = 'female'
gender_nonconformity = "nonconformity"

# Databases
database_ravdess = 'ravdess'
database_emodb = 'emodb'
database_cremad = 'cremad'
database_shemo = 'shemo'

# Datasets
dataset_ravdess_male = 'ravdess_m'
dataset_ravdess_female = 'ravdess_f'
dataset_emodb_male = 'emodb_m'
dataset_emodb_female = 'emodb_f'
dataset_cremad_male = 'cremad_m'
dataset_cremad_female = 'cremad_f'
dataset_shemo_male = 'shemo_m'
dataset_shemo_female = 'shemo_f'

# Database audio sample paths
main_databases_path = '../../../Resources/Databases'
smpl_data_path_ravdess = os.path.join(main_databases_path, "RAVDESS_Audio")
smpl_data_path_emodb = os.path.join(main_databases_path, "EmoDB", "wav")
smpl_data_path_cremad = os.path.join(main_databases_path, "CREMA-D", "AudioWAV")
info_path_cremad = os.path.join(main_databases_path, "CREMA-D", "VideoDemographics.csv")
smpl_data_path_shemo = os.path.join(main_databases_path, "ShEMO")

# Primary Emotions
N = "neutral"
C = "calm"
H = "happy"
S = "sad"
A = "angry"
F = "fearful"
D = "disgust"
Su = "surprised"
B = "bored"
Non = "none"

# Status
original = 'original'
clean = 'clean'

# Modes
mode_write = 'write'
mode_read = 'read'

# Cleaning Parameters
resample_rate = 16000
step_size = int(resample_rate/10)

# Model Modes
ml_mode_convolutional = 'convolutional'
