from backbone_independent.support import configurations_variables as confv
import os


'''
class ModelConfig:
    pass
'''


# The below ModelConfig is only for reference. Just the name of ModelConfig is enough as above.
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


class DictToValAccMap:
    def __init__(self, dict, val_acc):
        self.dict = dict
        self.val_acc = val_acc


class DatasetHelper:
    def __init__(self, modelconfig, loaded_model, val_acc):
        self.modelconfig = modelconfig
        self.loaded_model = loaded_model
        self.val_acc = val_acc
