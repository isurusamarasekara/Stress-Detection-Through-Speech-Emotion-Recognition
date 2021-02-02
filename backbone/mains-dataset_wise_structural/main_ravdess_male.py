import backbone.support.configurations_variables as confv
import backbone.support.data_loading as dl
import backbone.support.data_analysis as da
import backbone.support.data_cleaning as dc
import backbone.support.configuration_classes as confc
import backbone.support.saving_loading as sl
import backbone.support.plots_and_charts as pc
import backbone.support.build_features as bf
import numpy as np
import backbone.support.models as mdl
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import TensorBoard
import time
import backbone.support.directory_file_checking as dfc
import os
from tensorflow.python.keras.callbacks import CSVLogger
import tensorflow as tf

print("\t===========================================================================================\n"
      "\t\tMain program started for MAIN-DATABASE:{database}, GENDER-ISOLATION:{gender}\n"
      "\t\t\t\u2234 Dataset Name: {name}\n"
      "\t==========================================================================================="
      .format(database=confv.database_ravdess, gender=confv.gender_male, name=confv.dataset_ravdess_male))

'''
# DATA LOADING SECTION
print("\n--------------------Started loading original data from the main database: {name}--------------------".format(name=confv.database_ravdess))
data_info_ravdess_df = dl.load_original_data(database=confv.database_ravdess)
print("No. of sample audio files in {database} database: {length}\n".format(database=confv.database_ravdess, length=len(data_info_ravdess_df)))
print("Dataframe head of {database} database:".format(database=confv.database_ravdess))
print(data_info_ravdess_df.head())
print("\nDataframe tail of {database} database:".format(database=confv.database_ravdess))
print(data_info_ravdess_df.tail())
print("--------------------Finished loading original data from the main database: {name}--------------------".format(name=confv.database_ravdess))


# RANDOM BASE AUDIO WAVE ANALYSIS SECTION
print("\n\n--------------------Started random base audio wave analysis for the main database: {name}--------------------".format(name=confv.database_ravdess))
da.base_audio_wave_analysis(data_info_ravdess_df.audio_fname[500], database=confv.database_ravdess, status=confv.original)
print("--------------------Finished random base audio wave analysis for the main database: {name}--------------------".format(name=confv.database_ravdess))


# DATAFRAME ADJUSTMENTS SECTION
print("\n\n--------------------Started dataframe adjustment for the main database: {name}--------------------".format(name=confv.database_ravdess))
data_info_ravdess_df_m, data_info_ravdess_df_f = dc.data_adjustments(data_info_ravdess_df)
print("--------------------Finished dataframe adjustment for the main database: {name}--------------------".format(name=confv.database_ravdess))


# DATAFRAME SAVING
print("\n\n--------------------Started dataframe saving for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
ravdess_m_df_obj = confc.DataFrame(database=confv.database_ravdess, gender=confv.gender_male, df=data_info_ravdess_df_m)
sl.save_dataframe(ravdess_m_df_obj)
print("--------------------Finished dataframe saving for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
'''

# LOAD REQUIRED PICKLE
print("\n\n--------------------Started dataframe loading for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
ravdess_m_df_obj = confc.DataFrame(database=confv.database_ravdess, gender=confv.gender_male)
ravdess_m_df_obj = sl.load_dataframe(ravdess_m_df_obj)
data_info_ravdess_df_m = ravdess_m_df_obj.df
print(ravdess_m_df_obj.database)
print(ravdess_m_df_obj.gender)
print(len(data_info_ravdess_df_m))
print(data_info_ravdess_df_m.head())
print(data_info_ravdess_df_m.tail())
print(ravdess_m_df_obj.dataset)
print(ravdess_m_df_obj.save_path)
print("--------------------Finished dataframe loading for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))

'''
# ORIGINAL DATA DISTRIBUTION ANALYSIS SECTION
print("\n\n--------------------Started original data distribution analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
pc.emotion_distribution_bar_plot(df=data_info_ravdess_df_m, title="{database} - {gender} Isolation - No. of Files".format(database=confv.database_ravdess, gender=confv.gender_male))
pc.emotion_distribution_pie_plot(df=data_info_ravdess_df_m, database=confv.database_ravdess, status=confv.original, gender=confv.gender_male, title="{database} - {gender} Isolation - Class/Data/Time Distribution".format(database=confv.database_ravdess, gender=confv.gender_male))
print("--------------------Finished original data distribution analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))


# ORIGINAL DATA VISUAL ANALYSIS (signal, fft, fbank, mfcc) SECTION
print("\n\n--------------------Started original data visual analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
da.visual_analysis(df=data_info_ravdess_df_m, database=confv.database_ravdess, status=confv.original, gender=confv.gender_male, envelope=False, resample=False)
da.visual_analysis(df=data_info_ravdess_df_m, database=confv.database_ravdess, status=confv.original, gender=confv.gender_male, envelope=True, resample=True)
print("--------------------Finished original data visual analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))


# DATA CLEANING - DOWN SAMPLING AND NOISE FLOOR DETECTION
print("\n\n--------------------Started data cleaning for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
dc.data_cleaning(df=data_info_ravdess_df_m, database=confv.database_ravdess)
print("--------------------Finished data cleaning for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
'''

# DATA MINIMUM AUDIO LENGTH COMPLIANCE CHECK
print("\n\n--------------------Started data minimum audio compliance check for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
data_info_ravdess_df_m = dc.check_and_adjust_df_for_minimum_audio_length_after_cleaning(df=data_info_ravdess_df_m, database=confv.database_ravdess, gender=confv.gender_male)
print("--------------------Finished data minimum audio compliance check for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))

'''
# CLEANED DATA DISTRIBUTION ANALYSIS SECTION
print("\n\n--------------------Started cleaned data distribution analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
pc.emotion_distribution_bar_plot(df=data_info_ravdess_df_m, title="{database} - {gender} Isolation - No. of Files".format(database=confv.database_ravdess, gender=confv.gender_male))
pc.emotion_distribution_pie_plot(df=data_info_ravdess_df_m, database=confv.database_ravdess, status=confv.clean, gender=confv.gender_male, title="{database} - {gender} Isolation - Class/Data/Time Distribution".format(database=confv.database_ravdess, gender=confv.gender_male))
print("--------------------Finished cleaned data distribution analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))


# CLEANED DATA VISUAL ANALYSIS (signal, fft, fbank, mfcc) SECTION
print("\n\n--------------------Started cleaned data visual analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
da.visual_analysis(df=data_info_ravdess_df_m, database=confv.database_ravdess, status=confv.clean, gender=confv.gender_male, envelope=False, resample=False)
# This is same as,
# da.visual_analysis(df=data_info_ravdess_df_m, database=confv.database_ravdess, status=confv.original, gender=confv.gender_male, envelope=True, resample=True)
# Since these cleaned data are already equipped with envelope and resampling, setting them to False or True does not matter.
# (envelope and resample does not matter when its clean)
print("--------------------Finished cleaned data visual analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
'''

# Building Features
print("\n\n--------------------Started building features for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
classes = list(np.unique(data_info_ravdess_df_m.stress_emotion))
mconf_ravdess_m = confc.ModelConfig(database=confv.database_ravdess, gender=confv.gender_male, mode=confv.ml_mode_convolutional, classes=classes)
print(mconf_ravdess_m.database)
print(mconf_ravdess_m.gender)
print(mconf_ravdess_m.mode)
print(mconf_ravdess_m.nfilt)
print(mconf_ravdess_m.nfeat)
print(mconf_ravdess_m.nfft)
print(mconf_ravdess_m.step)
print(mconf_ravdess_m.classes)
print(mconf_ravdess_m.features_save_name)
print(mconf_ravdess_m.model_config_save_name)
print(mconf_ravdess_m.training_log_name)
print(mconf_ravdess_m.model_save_name)
print(mconf_ravdess_m.model_h5_save_name)
print(mconf_ravdess_m.model_tflite_save_name)
print(mconf_ravdess_m.feature_path)
print(mconf_ravdess_m.model_config_path)
print(mconf_ravdess_m.training_log_path)
print(mconf_ravdess_m.model_path)
print(mconf_ravdess_m.model_h5_path)
print(mconf_ravdess_m.model_tflite_path)
rfpconf_ravdess_m = confc.RandFeatParams(df=data_info_ravdess_df_m, database=confv.database_ravdess, gender=confv.gender_male)
X, y = bf.build_random_features(modelconfig=mconf_ravdess_m, randfeatparams=rfpconf_ravdess_m)
print("--------------------Finished building features for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))


# MODEL & TRAINING
print("\n\n--------------------Started model training for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
input_shape = (X.shape[1], X.shape[2], 1)
model = mdl.get_ravdess_male_model(input_shape)

y_flat = np.argmax(y, axis=1)
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
class_weight = {i : class_weight[i] for i in range(2)}

NAME = "{database}-{gender}-{modeltype}-{spec}-{time}".format(database=confv.database_ravdess, gender=confv.gender_male, modeltype=confv.ml_mode_convolutional, spec="1st", time=int(time.time()))
mdl_logs_pth = os.path.join(confv.base_store, confv.log_dir)
tensorboard = TensorBoard(log_dir=mdl_logs_pth + '\\{}'.format(NAME))

dfc.check_dir_inside_saved_features_and_modelconfigs_and_models(parent=confv.saved_training_metrics_logs, database=confv.database_ravdess, gender=confv.gender_male)
csv_logger = CSVLogger(mconf_ravdess_m.training_log_path)
model.fit(X, y, epochs=35, batch_size=128, shuffle=True, class_weight=class_weight, validation_split=0.2, callbacks=[tensorboard, csv_logger])
# Can improve... more epochs
print("--------------------Finished model training for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))


# MODEL SAVING
print("\n\n--------------------Started model saving for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
dfc.check_dir_inside_saved_features_and_modelconfigs_and_models(parent=confv.saved_models, database=confv.database_ravdess, gender=confv.gender_male)
model.save(mconf_ravdess_m.model_path)
model.save(mconf_ravdess_m.model_h5_path)

# Convert the model & save in tflite
converter = tf.lite.TFLiteConverter.from_saved_model(mconf_ravdess_m.model_path)      # path to the SavedModel directory
tflite_model = converter.convert()
with open(mconf_ravdess_m.model_tflite_path, 'wb') as outfile:
    outfile.write(tflite_model)
print("--------------------Finished model saving for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_ravdess_male))
