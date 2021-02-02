from speech_analysis_raspi.support import saving_loading as sl
from speech_analysis_raspi.support import configurations_variables as confv
from speech_analysis_raspi.support import configurations_methods as confm
from speech_analysis_raspi.support import directory_file_checking as dfc
import sys
import tflite_runtime.interpreter as tflite
from speech_analysis_raspi.support import calculations as calc
from speech_analysis_raspi.support import configuration_classes as confc
from speech_analysis_raspi.support import  recording_configurations as rconf
from python_speech_features import mfcc
import numpy as np


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

    model_tflite_path = confm.get_universal_path(mconf.model_tflite_path)
    if not dfc.check_file_existence(model_tflite_path):
        print("Saved model does not exist for {database} - {gender}".format(database=database, gender=gender))
        sys.exit()

    # TFLite - Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_tflite_path)

    # Get validation accuracy
    val_acc = calc.get_validation_accuracy(confm.get_universal_path(mconf.training_log_path))

    # Creating dataset_helper to hold respective ModelConfig and interpreter objects with validation accuracies
    dataset_helper = confc.DatasetHelper(modelconfig=mconf, interpreter=interpreter, val_acc=val_acc)

    return dataset_helper


def predict_real_time(dataset_helpers_dict, signal):
    dict_act_map_list = []
    for db, helper in dataset_helpers_dict.items():
        # print(db, helper, helper.modelconfig, helper.interpreter, helper.val_acc)
        dict_act_map = predict(helper=helper, signal=signal)
        # print('dict_act_map', dict_act_map)
        # print('dict_act_map.dict', dict_act_map.dict, 'dict_act_map.val_acc', dict_act_map.val_acc)
        dict_act_map_list.append(dict_act_map)
        # print(dict_act_map_list)

    final_result = calc.get_biased_probs(dict_act_map_list)
    # print('final_result', final_result)

    return final_result


def predict(helper: confc.DatasetHelper, signal):
    # TFLite - Load the TFLite model and allocate tensors.
    interpreter = helper.interpreter
    interpreter.allocate_tensors()

    # TFLite - Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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

        # TFLite - convert array to relevant datatype noted in input_details[0]['dtype']
        x = np.array(x, dtype=np.float32)
        # TFLite - set the value of the input tensor
        interpreter.set_tensor(input_details[0]['index'], x)
        # TFLite - invoke the interpreter
        interpreter.invoke()
        # TFLite - get the value of the input tensor
        y_hat = interpreter.get_tensor(output_details[0]['index'])

        y_prob.append(y_hat)

    fn_prob = np.mean(y_prob, axis=0).flatten()
    # print("The mean of all probabilities of", count, "1/10 second chunks from the single audio file for the classes ", mconf.classes, " are ", fn_prob)

    results_dict = calc.get_dict(mconf.classes, fn_prob)
    # print('results_dict', results_dict)

    dict_act_map = confc.DictToValAccMap(dict=results_dict, val_acc=helper.val_acc)
    # print('dict_act_map', dict_act_map)
    # print('dict_act_map.dict', dict_act_map.dict, 'dict_act_map.val_acc', dict_act_map.val_acc)

    return dict_act_map
