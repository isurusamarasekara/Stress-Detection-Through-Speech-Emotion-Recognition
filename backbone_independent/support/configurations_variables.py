import os

# Genders
gender_male = 'male'
gender_female = 'female'

# Databases
database_ravdess = 'ravdess'
database_emodb = 'emodb'
database_cremad = 'cremad'
database_shemo = 'shemo'

# Model Modes
ml_mode_convolutional = 'convolutional'

# Support Directories
base_store = '../base_store'
saved_features = 'saved_features'   # Not required - Only included due to error shown in configuration_classes/ModelConfig reference class
saved_modelconfigs = 'saved_modelconfigs'
saved_training_metrics_logs = 'saved_training_metrics_logs'   # Not required - Only included due to error shown in configuration_classes/ModelConfig reference class
saved_models = 'saved_models'      # Not required - Only included due to error shown in configuration_classes/ModelConfig reference class

# Pre-prediction store
preprediction_base_audio_store = '../preprediction_audio_store'
prerecorded_upload_dir = 'uploads'
prerecorded_upload_dir_pth = os.path.join(preprediction_base_audio_store, prerecorded_upload_dir)

# Upload and time-specified-prerecorded file names and paths
prerecorded_upload_file_name = 'prerecordedUpload.wav'
prerecorded_upload_file_aud_fl_pth = os.path.join(prerecorded_upload_dir_pth, prerecorded_upload_file_name)

# Sampling Parameters - The below two variables cannot change unless model training changes
resample_rate = 16000
step_size = int(resample_rate/10)   # 10 cannot change unless modelconfig.step changes,
                                    # This is bound to build features in model training.

# Envelope thresholds - mostly like loudness threshold.
prerecorded_upload_envelope_threshold = 0.003   # 0.034 is better for new uploads. 0.003 for files from databases
real_time_noisy_envelope_threshold = 0.0005

# Classes
class_stressed = 'Stressed'
class_not_stressed = 'Not Stressed'

# Application types
app_type_prerecorded_upload = 'prerecorded_upload'
app_type_real_time = 'realtime'

