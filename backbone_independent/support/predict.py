from backbone_independent.support import configurations_variables as confv
from backbone_independent.support import saving_loading as sl
from keras.models import load_model
from python_speech_features import mfcc
import numpy as np
from backbone_independent.support import calculations as calc
from backbone_independent.support import configurations_methods as confm
from backbone_independent.support import directory_file_checking as dfc
import sys
from backbone_independent.support import configuration_classes as confc
from backbone_independent.support import recording_configurations as rconf


def predict_upload(signal, gender):
    rate = confv.resample_rate
    if gender == confv.gender_male:
        final_result = predict_male(signal, rate)
    elif gender == confv.gender_female:
        final_result = predict_female(signal, rate)

    return final_result


def predict_male(signal, rate):
    male_dict_act_map_list = []

    ravdess_male_dict_act_map = predict_common(database=confv.database_ravdess, gender=confv.gender_male, mode=confv.ml_mode_convolutional, signal=signal, rate=rate)
    male_dict_act_map_list.append(ravdess_male_dict_act_map)

    # emodb_male_dict_act_map = predict_common(database=confv.database_emodb, gender=confv.gender_male, mode=confv.ml_mode_convolutional, signal=signal, rate=rate)
    # male_dict_act_map_list.append(emodb_male_dict_act_map)
    #
    # cremad_male_dict_act_map = predict_common(database=confv.database_cremad, gender=confv.gender_male, mode=confv.ml_mode_convolutional, signal=signal, rate=rate)
    # male_dict_act_map_list.append(cremad_male_dict_act_map)

    shemo_male_dict_act_map = predict_common(database=confv.database_shemo, gender=confv.gender_male, mode=confv.ml_mode_convolutional, signal=signal, rate=rate)
    male_dict_act_map_list.append(shemo_male_dict_act_map)

    final_result_male = calc.get_biased_probs(male_dict_act_map_list)

    return final_result_male


def predict_female(signal, rate):
    female_dict_act_map_list = []

    ravdess_female_dict_act_map = predict_common(database=confv.database_ravdess, gender=confv.gender_female, mode=confv.ml_mode_convolutional, signal=signal, rate=rate)
    female_dict_act_map_list.append(ravdess_female_dict_act_map)

    # emodb_female_dict_act_map = predict_common(database=confv.database_emodb, gender=confv.gender_female, mode=confv.ml_mode_convolutional, signal=signal, rate=rate)
    # female_dict_act_map_list.append(emodb_female_dict_act_map)

    cremad_female_dict_act_map = predict_common(database=confv.database_cremad, gender=confv.gender_female, mode=confv.ml_mode_convolutional, signal=signal, rate=rate)
    female_dict_act_map_list.append(cremad_female_dict_act_map)

    shemo_female_dict_act_map = predict_common(database=confv.database_shemo, gender=confv.gender_female, mode=confv.ml_mode_convolutional, signal=signal, rate=rate)
    female_dict_act_map_list.append(shemo_female_dict_act_map)

    final_result_female = calc.get_biased_probs(female_dict_act_map_list)

    return final_result_female


def predict_common(database, gender, mode, signal, rate):
    mconf = sl.load_model_config(database=database, gender=gender, mode=mode)
    # print(mconf.database)
    # print(mconf.gender)
    # print(mconf.mode)
    # print(mconf.nfilt)
    # print(mconf.nfeat)
    # print(mconf.nfft)
    # print(mconf.step)
    # print(mconf.classes)
    # print(mconf.features_save_name)
    # print(mconf.model_config_save_name)
    # print(mconf.training_log_name)
    # print(mconf.model_save_name)
    # print(mconf.model_h5_save_name)
    # print(mconf.model_tflite_save_name)
    # print(mconf.feature_path)
    # print(mconf.model_config_path)
    # print(mconf.training_log_path)
    # print(mconf.model_path)
    # print(mconf.model_h5_path)
    # print(mconf.model_tflite_path)

    model_path = confm.get_universal_path(path=mconf.model_h5_path)
    if not dfc.check_file_existence(model_path):
        print("Saved model does not exist for {database} - {gender}".format(database=mconf.database, gender=mconf.gender))
        sys.exit()

    loaded_model = load_model(model_path)  # No need to use the model path from modelconfig. B/c this restricts me to
    # the exact same saved_models directory structure as used in the backbone development.
    # Just can get the path from using os.path.join

    y_pred = []
    y_prob = []
    count = 0
    for i in range(0, signal.shape[0] - mconf.step, mconf.step):
        count = count + 1   # Which 10th of a second number
        sample = signal[i:i + mconf.step]
        x = mfcc(sample, rate, numcep=mconf.nfeat, nfilt=mconf.nfilt, nfft=mconf.nfft)
        x = (x - mconf.min) / (mconf.max - mconf.min)

        x = x.reshape(1, x.shape[0], x.shape[1], 1)

        y_hat = loaded_model.predict(x)
        # print("10th of a second chunk number: ", count)
        # print("Probabilities of prediction for the classes ", mconf.classes, " are, respectively, ", y_hat)
        # print("Relevant final prediction for the #", count, " 1/10 second chunk: ", [mconf.classes[np.argmax(y)] for y in y_hat], '\n')

        y_prob.append(y_hat)

    fn_prob = np.mean(y_prob, axis=0).flatten()
    print("{database}-{gender}: The mean of all probabilities of {count} 1/10 second chunks from the single audio file for the classes {classes} are {probs}".format(database=mconf.database, gender=mconf.gender, count=count, classes=mconf.classes, probs=fn_prob))

    results_dict = calc.get_dict(classes=mconf.classes, probs=fn_prob)
    val_acc = calc.get_validation_accuracy(training_log_path=confm.get_universal_path(mconf.training_log_path))

    dict_act_map = confc.DictToValAccMap(dict=results_dict, val_acc=val_acc)

    return dict_act_map


def load_all_helpers_dict(database_dict, gender):
    # print(database_dict)

    dataset_helpers_dict = {}
    for db, mode in database_dict.items():
        # print(db, mode)
        dataset_helpers_dict[db] = get_helper(database=db, gender=gender, mode=mode)

    return dataset_helpers_dict


def get_helper(database, gender, mode):
    mconf = sl.load_model_config(database=database, gender=gender, mode=mode)
    # print(mconf.database)
    # print(mconf.gender)
    # print(mconf.mode)
    # print(mconf.nfilt)
    # print(mconf.nfeat)
    # print(mconf.nfft)
    # print(mconf.step)
    # print(mconf.classes)
    # print(mconf.features_save_name)
    # print(mconf.model_config_save_name)
    # print(mconf.training_log_name)
    # print(mconf.model_save_name)
    # print(mconf.model_h5_save_name)
    # print(mconf.model_tflite_save_name)
    # print(mconf.feature_path)
    # print(mconf.model_config_path)
    # print(mconf.training_log_path)
    # print(mconf.model_path)
    # print(mconf.model_h5_path)
    # print(mconf.model_tflite_path)

    model_path = confm.get_universal_path(path=mconf.model_h5_path)
    if not dfc.check_file_existence(model_path):
        print("Saved model does not exist for {database} - {gender}".format(database=mconf.database, gender=mconf.gender))
        sys.exit()

    # Load the HDF5 model.
    # Or even Keras SavedModel
    loaded_model = load_model(model_path)

    # Get validation accuracy
    val_acc = calc.get_validation_accuracy(confm.get_universal_path(mconf.training_log_path))

    # Creating dataset_helper to hold respective ModelConfig and interpreter objects with validation accuracies
    dataset_helper = confc.DatasetHelper(modelconfig=mconf, loaded_model=loaded_model, val_acc=val_acc)

    return dataset_helper


def predict_real_time(dataset_helpers_dict, signal):
    dict_act_map_list = []
    for db, helper in dataset_helpers_dict.items():
        # print(db, helper, helper.modelconfig, helper.interpreter, helper.val_acc)
        dict_act_map = predict_real_time_common(helper=helper, signal=signal)
        # print('dict_act_map', dict_act_map)
        # print('dict_act_map.dict', dict_act_map.dict, 'dict_act_map.val_acc', dict_act_map.val_acc)
        dict_act_map_list.append(dict_act_map)
        # print(dict_act_map_list)

    final_result = calc.get_biased_probs(dict_act_map_list)
    # print('final_result', final_result)

    return final_result


def predict_real_time_common(helper: confc.DatasetHelper, signal):
    loaded_model = helper.loaded_model
    mconf = helper.modelconfig
    y_pred = []
    y_prob = []
    count = 0
    for i in range(0, signal.shape[0] - mconf.step, mconf.step):
        count = count + 1   # Which 10th of a second number
        sample = signal[i:i + mconf.step]
        x = mfcc(sample, rconf.RATE_realtime, numcep=mconf.nfeat, nfilt=mconf.nfilt, nfft=mconf.nfft)
        x = (x - mconf.min) / (mconf.max - mconf.min)

        x = x.reshape(1, x.shape[0], x.shape[1], 1)

        y_hat = loaded_model.predict(x)

        y_prob.append(y_hat)

    fn_prob = np.mean(y_prob, axis=0).flatten()
    # print("The mean of all probabilities of", count, "1/10 second chunks from the single audio file for the classes ", mconf.classes, " are ", fn_prob)

    results_dict = calc.get_dict(mconf.classes, fn_prob)
    # print('results_dict', results_dict)

    dict_act_map = confc.DictToValAccMap(dict=results_dict, val_acc=helper.val_acc)
    # print('dict_act_map', dict_act_map)
    # print('dict_act_map.dict', dict_act_map.dict, 'dict_act_map.val_acc', dict_act_map.val_acc)

    return dict_act_map