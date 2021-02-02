import backbone.support.configurations_variables as confv
import os
import numpy as np
import backbone.support.data_loading as dl


class DataFrame:
    database = ""
    gender = ""
    df = ""
    dataset = ""
    save_path = ""

    def __init__(self, database, gender, df=""):
        self.database = database
        self.gender = gender
        self.df = df
        if database == confv.database_ravdess and gender == confv.gender_male:
            self.dataset = confv.dataset_ravdess_male
        elif database == confv.database_ravdess and gender == confv.gender_female:
            self.dataset = confv.dataset_ravdess_female
        elif database == confv.database_emodb and gender == confv.gender_male:
            self.dataset = confv.dataset_emodb_male
        elif database == confv.database_emodb and gender == confv.gender_female:
            self.dataset = confv.dataset_emodb_female
        elif database == confv.database_cremad and gender == confv.gender_male:
            self.dataset = confv.dataset_cremad_male
        elif database == confv.database_cremad and gender == confv.gender_female:
            self.dataset = confv.dataset_cremad_female
        elif database == confv.database_shemo and gender == confv.gender_male:
            self.dataset = confv.dataset_shemo_male
        elif database == confv.database_shemo and gender == confv.gender_female:
            self.dataset = confv.dataset_shemo_female

        self.save_path = os.path.join(confv.base_store, confv.saved_dataframes, self.dataset)


class ModelConfig:
    def __init__(self, database, gender, mode, classes='', nfilt=26, nfeat=13, nfft=512):
        self.database = database
        self.gender = gender
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.step = confv.step_size     # Make sure this is integer
        self.classes = classes

        # self.model_config_save_name = ""

        self.features_save_name = self.mode + ".p"
        self.model_config_save_name = self.mode
        self.training_log_name = self.mode + "_training.log"
        self.model_save_name = self.mode + ".model"     # SavedModel
        self.model_h5_save_name = self.mode + ".h5"     # HDF5
        self.model_tflite_save_name = self.mode + ".tflite"

        # if dataset == dataset_ravdess and gender == gender_male:
        #     self.model_config_save_name = clean_dir_ravdess_m
        # elif dataset == dataset_ravdess and gender == gender_female:
        #     self.model_config_save_name = clean_dir_ravdess_f
        # elif dataset == dataset_emodb and gender == gender_male:
        #     self.model_config_save_name = clean_dir_emodb_m
        # elif dataset == dataset_emodb and gender == gender_female:
        #     self.model_config_save_name = clean_dir_emodb_f
        # elif dataset == dataset_cremad and gender == gender_male:
        #     self.model_config_save_name = clean_dir_cremad_m
        # elif dataset == dataset_cremad and gender == gender_female:
        #     self.model_config_save_name = clean_dir_cremad_f
        # elif dataset == dataset_shemo and gender == gender_male:
        #     self.model_config_save_name = clean_dir_shemo_m
        # elif dataset == dataset_shemo and gender == gender_female:
        #     self.model_config_save_name = clean_dir_shemo_f

        # Dont need feature_path in future preds
        self.feature_path = os.path.join(confv.base_store, confv.saved_features, self.database, self.gender, self.features_save_name)
        self.model_config_path = os.path.join(confv.base_store, confv.saved_modelconfigs, self.database, self.gender, self.model_config_save_name)
        self.training_log_path = os.path.join(confv.base_store, confv.saved_training_metrics_logs, self.database, self.gender, self.training_log_name)
        self.model_path = os.path.join(confv.base_store, confv.saved_models, self.database, self.gender, self.model_save_name)
        self.model_h5_path = os.path.join(confv.base_store, confv.saved_models, self.database, self.gender, self.model_h5_save_name)
        self.model_tflite_path = os.path.join(confv.base_store, confv.saved_models, self.database, self.gender, self.model_tflite_save_name)


class FeatureSet:
    def __init__(self, X, y):
        self.X = X
        self.y = y


class RandFeatParams:
    def __init__(self, df, database, gender):
        self.df2 = df.copy()
        self.df1 = dl.get_df_with_length(df=self.df2, database=database, status=confv.clean, gender=gender)
        self.n_samples = 2 * int(self.df1['length'].sum() / 0.1)    # Static ?
        self.class_dist = self.df1.groupby(['stress_emotion'])['length'].mean()
        self.prob_dist = self.class_dist / self.class_dist.sum()
        self.classes = list(np.unique(self.df1.stress_emotion))
