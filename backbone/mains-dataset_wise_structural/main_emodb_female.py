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
      .format(database=confv.database_emodb, gender=confv.gender_female, name=confv.dataset_emodb_female))

'''
# DATA LOADING SECTION
print("\n--------------------Started loading original data from the main database: {name}--------------------".format(name=confv.database_emodb))
data_info_emodb_df = dl.load_original_data(database=confv.database_emodb)
print("No. of sample audio files in {database} database: {length}\n".format(database=confv.database_emodb, length=len(data_info_emodb_df)))
print("Dataframe head of {database} database:".format(database=confv.database_emodb))
print(data_info_emodb_df.head())
print("\nDataframe tail of {database} database:".format(database=confv.database_emodb))
print(data_info_emodb_df.tail())
print("--------------------Finished loading original data from the main database: {name}--------------------".format(name=confv.database_emodb))


# RANDOM BASE AUDIO WAVE ANALYSIS SECTION
print("\n\n--------------------Started random base audio wave analysis for the main database: {name}--------------------".format(name=confv.database_emodb))
da.base_audio_wave_analysis(data_info_emodb_df.audio_fname[500], database=confv.database_emodb, status=confv.original)
print("--------------------Finished random base audio wave analysis for the main database: {name}--------------------".format(name=confv.database_emodb))


# DATAFRAME ADJUSTMENTS SECTION
print("\n\n--------------------Started dataframe adjustment for the main database: {name}--------------------".format(name=confv.database_emodb))
data_info_emodb_df_m, data_info_emodb_df_f = dc.data_adjustments(data_info_emodb_df)
print("--------------------Finished dataframe adjustment for the main database: {name}--------------------".format(name=confv.database_emodb))


# DATAFRAME SAVING
print("\n\n--------------------Started dataframe saving for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
emodb_f_df_obj = confc.DataFrame(database=confv.database_emodb, gender=confv.gender_female, df=data_info_emodb_df_f)
sl.save_dataframe(emodb_f_df_obj)
print("--------------------Finished dataframe saving for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
'''

# LOAD REQUIRED PICKLE
print("\n\n--------------------Started dataframe loading for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
emodb_f_df_obj = confc.DataFrame(database=confv.database_emodb, gender=confv.gender_female)
emodb_f_df_obj = sl.load_dataframe(emodb_f_df_obj)
data_info_emodb_df_f = emodb_f_df_obj.df
print(emodb_f_df_obj.database)
print(emodb_f_df_obj.gender)
print(len(data_info_emodb_df_f))
print(data_info_emodb_df_f.head())
print(data_info_emodb_df_f.tail())
print(emodb_f_df_obj.dataset)
print(emodb_f_df_obj.save_path)
print("--------------------Finished dataframe loading for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))

'''
# ORIGINAL DATA DISTRIBUTION ANALYSIS SECTION
print("\n\n--------------------Started original data distribution analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
pc.emotion_distribution_bar_plot(df=data_info_emodb_df_f, title="{database} - {gender} Isolation - No. of Files".format(database=confv.database_emodb, gender=confv.gender_female))
pc.emotion_distribution_pie_plot(df=data_info_emodb_df_f, database=confv.database_emodb, status=confv.original, gender=confv.gender_female, title="{database} - {gender} Isolation - Class/Data/Time Distribution".format(database=confv.database_emodb, gender=confv.gender_female))
print("--------------------Finished original data distribution analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))


# ORIGINAL DATA VISUAL ANALYSIS (signal, fft, fbank, mfcc) SECTION
print("\n\n--------------------Started original data visual analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
da.visual_analysis(df=data_info_emodb_df_f, database=confv.database_emodb, status=confv.original, gender=confv.gender_female, envelope=False, resample=False)
da.visual_analysis(df=data_info_emodb_df_f, database=confv.database_emodb, status=confv.original, gender=confv.gender_female, envelope=True, resample=True)
print("--------------------Finished original data visual analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))


# DATA CLEANING - DOWN SAMPLING AND NOISE FLOOR DETECTION
print("\n\n--------------------Started data cleaning for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
dc.data_cleaning(df=data_info_emodb_df_f, database=confv.database_emodb)
print("--------------------Finished data cleaning for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
'''

# DATA MINIMUM AUDIO LENGTH COMPLIANCE CHECK
print("\n\n--------------------Started data minimum audio compliance check for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
data_info_emodb_df_f = dc.check_and_adjust_df_for_minimum_audio_length_after_cleaning(df=data_info_emodb_df_f, database=confv.database_emodb, gender=confv.gender_female)
print("--------------------Finished data minimum audio compliance check for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))

'''
# CLEANED DATA DISTRIBUTION ANALYSIS SECTION
print("\n\n--------------------Started cleaned data distribution analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
pc.emotion_distribution_bar_plot(df=data_info_emodb_df_f, title="{database} - {gender} Isolation - No. of Files".format(database=confv.database_emodb, gender=confv.gender_female))
pc.emotion_distribution_pie_plot(df=data_info_emodb_df_f, database=confv.database_emodb, status=confv.clean, gender=confv.gender_female, title="{database} - {gender} Isolation - Class/Data/Time Distribution".format(database=confv.database_emodb, gender=confv.gender_female))
print("--------------------Finished cleaned data distribution analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))


# CLEANED DATA VISUAL ANALYSIS (signal, fft, fbank, mfcc) SECTION
print("\n\n--------------------Started cleaned data visual analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
da.visual_analysis(df=data_info_emodb_df_f, database=confv.database_emodb, status=confv.clean, gender=confv.gender_female, envelope=False, resample=False)
# This is same as,
# da.visual_analysis(df=data_info_emodb_df_f, database=confv.database_emodb, status=confv.original, gender=confv.gender_female, envelope=True, resample=True)
# Since these cleaned data are already equipped with envelope and resampling, setting them to False or True does not matter.
# (envelope and resample does not matter when its clean)
print("--------------------Finished cleaned data visual analysis for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
'''

# Building Features
print("\n\n--------------------Started building features for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
classes = list(np.unique(data_info_emodb_df_f.stress_emotion))
mconf_emodb_f = confc.ModelConfig(database=confv.database_emodb, gender=confv.gender_female, mode=confv.ml_mode_convolutional, classes=classes)
print(mconf_emodb_f.database)
print(mconf_emodb_f.gender)
print(mconf_emodb_f.mode)
print(mconf_emodb_f.nfilt)
print(mconf_emodb_f.nfeat)
print(mconf_emodb_f.nfft)
print(mconf_emodb_f.step)
print(mconf_emodb_f.classes)
print(mconf_emodb_f.features_save_name)
print(mconf_emodb_f.model_config_save_name)
print(mconf_emodb_f.training_log_name)
print(mconf_emodb_f.model_save_name)
print(mconf_emodb_f.model_h5_save_name)
print(mconf_emodb_f.model_tflite_save_name)
print(mconf_emodb_f.feature_path)
print(mconf_emodb_f.model_config_path)
print(mconf_emodb_f.training_log_path)
print(mconf_emodb_f.model_path)
print(mconf_emodb_f.model_h5_path)
print(mconf_emodb_f.model_tflite_path)
rfpconf_emodb_f = confc.RandFeatParams(df=data_info_emodb_df_f, database=confv.database_emodb, gender=confv.gender_female)
X, y = bf.build_random_features(modelconfig=mconf_emodb_f, randfeatparams=rfpconf_emodb_f)
print("--------------------Finished building features for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))


# MODEL & TRAINING
print("\n\n--------------------Started model training for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_male, name=confv.dataset_emodb_male))
input_shape = (X.shape[1], X.shape[2], 1)
model = mdl.get_emodb_male_model(input_shape)

y_flat = np.argmax(y, axis=1)
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
class_weight = {i : class_weight[i] for i in range(2)}

NAME = "{database}-{gender}-{modeltype}-{spec}-{time}".format(database=confv.database_emodb, gender=confv.gender_female, modeltype=confv.ml_mode_convolutional, spec="1st", time=int(time.time()))
mdl_logs_pth = os.path.join(confv.base_store, confv.log_dir)
tensorboard = TensorBoard(log_dir=mdl_logs_pth + '\\{}'.format(NAME))

dfc.check_dir_inside_saved_features_and_modelconfigs_and_models(parent=confv.saved_training_metrics_logs, database=confv.database_emodb, gender=confv.gender_female)
csv_logger = CSVLogger(mconf_emodb_f.training_log_path)
model.fit(X, y, epochs=35, batch_size=128, shuffle=True, class_weight=class_weight, validation_split=0.2, callbacks=[tensorboard, csv_logger])
print("--------------------Finished model training for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))


# MODEL SAVING
print("\n\n--------------------Started model saving for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
dfc.check_dir_inside_saved_features_and_modelconfigs_and_models(parent=confv.saved_models, database=confv.database_emodb, gender=confv.gender_female)
model.save(mconf_emodb_f.model_path)
model.save(mconf_emodb_f.model_h5_path)

# Convert the model & save in tflite
converter = tf.lite.TFLiteConverter.from_saved_model(mconf_emodb_f.model_path)
tflite_model = converter.convert()
with open(mconf_emodb_f.model_tflite_path, 'wb') as outfile:
    outfile.write(tflite_model)
print("--------------------Finished model saving for adjusted and {gender} isolated dataset: {name}--------------------".format(gender=confv.gender_female, name=confv.dataset_emodb_female))
