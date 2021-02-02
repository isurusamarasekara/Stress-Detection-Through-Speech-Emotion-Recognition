from speech_analysis_raspi.support import configurations_variables as confv
from speech_analysis_raspi.support import calculations as calc
from speech_analysis_raspi.support import saving_loading as sl
from speech_analysis_raspi.support import directory_file_checking as dfc
from speech_analysis_raspi.support import configurations_methods as confm
import sys
import tflite_runtime.interpreter as tflite
from python_speech_features import mfcc
import numpy as np
from speech_analysis_raspi.support import configuration_classes as confc


'''
Can do the same as in realtime_predict where the databases and the modes will be defined at the main method, 
and can be controlled there easily.
This is just left as it is due to my own need of reflection
'''
def predict_upload_and_time_specified(signal, gender):
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

    final_result = calc.get_biased_probs(female_dict_act_map_list)

    return final_result


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

    model_path = confm.get_universal_path(mconf.model_tflite_path)
    if not dfc.check_file_existence(model_path):
        print("Saved model does not exist for {database} - {gender}".format(database=mconf.database, gender=mconf.gender))
        sys.exit()

    # TFLite - Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TFLite - Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    y_prob = []
    count = 0
    for i in range(0, signal.shape[0] - mconf.step, mconf.step):
        count = count + 1  # Which 10th of a second number
        sample = signal[i:i + mconf.step]
        x = mfcc(sample, rate, numcep=mconf.nfeat, nfilt=mconf.nfilt, nfft=mconf.nfft)
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
    print("{database}-{gender}: The mean of all probabilities of {count} 1/10 second chunks from the single audio file for the classes {classes} are {probs}".format(database=mconf.database, gender=mconf.gender, count=count, classes=mconf.classes, probs=fn_prob))

    results_dict = calc.get_dict(mconf.classes, fn_prob)
    val_acc = calc.get_validation_accuracy(confm.get_universal_path(mconf.training_log_path))

    dict_act_map = confc.DictToValAccMap(dict=results_dict, val_acc=val_acc)

    return dict_act_map

'''
def predict_ravdess_male(signal='', rate=''):
    mconf_ravdess_m = sl.load_model_config(database=confv.database_ravdess, gender=confv.gender_male, mode=confv.ml_mode_convolutional)
    # print(mconf_ravdess_m.database)
    # print(mconf_ravdess_m.gender)
    # print(mconf_ravdess_m.mode)
    # print(mconf_ravdess_m.nfilt)
    # print(mconf_ravdess_m.nfeat)
    # print(mconf_ravdess_m.nfft)
    # print(mconf_ravdess_m.step)
    # print(mconf_ravdess_m.classes)
    # print(mconf_ravdess_m.features_save_name)
    # print(mconf_ravdess_m.model_config_save_name)
    # print(mconf_ravdess_m.training_log_name)
    # print(mconf_ravdess_m.model_save_name)
    # print(mconf_ravdess_m.model_h5_save_name)
    # print(mconf_ravdess_m.model_tflite_save_name)
    # print(mconf_ravdess_m.feature_path)
    # print(mconf_ravdess_m.model_config_path)
    # print(mconf_ravdess_m.training_log_path)
    # print(mconf_ravdess_m.model_path)
    # print(mconf_ravdess_m.model_h5_path)
    # print(mconf_ravdess_m.model_tflite_path)

    model_path = confm.get_universal_path(mconf_ravdess_m.model_tflite_path)
    if not dfc.check_file_existence(model_path):
        print("Saved model does not exist for {database} - {gender}".format(database=confv.database_ravdess, gender=confv.gender_male))
        sys.exit()

    # TFLite - Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TFLite - Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    y_prob = []
    count = 0
    for i in range(0, signal.shape[0] - mconf_ravdess_m.step, mconf_ravdess_m.step):
        count = count + 1   # Which 10th of a second number
        sample = signal[i:i + mconf_ravdess_m.step]
        x = mfcc(sample, rate, numcep=mconf_ravdess_m.nfeat, nfilt=mconf_ravdess_m.nfilt, nfft=mconf_ravdess_m.nfft)
        x = (x - mconf_ravdess_m.min) / (mconf_ravdess_m.max - mconf_ravdess_m.min)

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
    print("{database}-{gender}: The mean of all probabilities of {count} 1/10 second chunks from the single audio file for the classes {classes} are {probs}".format(database=mconf_ravdess_m.database, gender=mconf_ravdess_m.gender, count=count, classes=mconf_ravdess_m.classes, probs=fn_prob))

    ravdess_male_results_dict = calc.get_dict(mconf_ravdess_m.classes, fn_prob)
    ravdess_male_val_acc = calc.get_validation_accuracy(confm.get_universal_path(mconf_ravdess_m.training_log_path))

    ravdess_male_dict_act_map = confc.DictToValAccMap(dict=ravdess_male_results_dict, val_acc=ravdess_male_val_acc)

    return ravdess_male_dict_act_map


def predict_emodb_male(signal='', rate=''):
    mconf_emodb_m = sl.load_model_config(database=confv.database_emodb, gender=confv.gender_male, mode=confv.ml_mode_convolutional)

    model_path = confm.get_universal_path(mconf_emodb_m.model_tflite_path)
    if not dfc.check_file_existence(model_path):
        print("Saved model does not exist for {database} - {gender}".format(database=confv.database_emodb, gender=confv.gender_male))
        sys.exit()

    # TFLite - Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TFLite - Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    y_prob = []
    count = 0
    for i in range(0, signal.shape[0] - mconf_emodb_m.step, mconf_emodb_m.step):
        count = count + 1  # Which 10th of a second number
        sample = signal[i:i + mconf_emodb_m.step]
        x = mfcc(sample, rate, numcep=mconf_emodb_m.nfeat, nfilt=mconf_emodb_m.nfilt, nfft=mconf_emodb_m.nfft)
        x = (x - mconf_emodb_m.min) / (mconf_emodb_m.max - mconf_emodb_m.min)

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
        y_pred.append(np.argmax(y_hat))

    fn_prob = np.mean(y_prob, axis=0).flatten()
    print("{database}-{gender}: The mean of all probabilities of {count} 1/10 second chunks from the single audio file for the classes {classes} are {probs}".format(database=mconf_emodb_m.database, gender=mconf_emodb_m.gender, count=count, classes=mconf_emodb_m.classes, probs=fn_prob))

    emodb_male_results_dict = calc.get_dict(mconf_emodb_m.classes, fn_prob)
    emodb_male_val_acc = calc.get_validation_accuracy(confm.get_universal_path(mconf_emodb_m.training_log_path))

    emodb_male_dict_act_map = confc.DictToValAccMap(dict=emodb_male_results_dict, val_acc=emodb_male_val_acc)

    return emodb_male_dict_act_map


def predict_cremad_male(signal='', rate=''):
    mconf_cremad_m = sl.load_model_config(database=confv.database_cremad, gender=confv.gender_male, mode=confv.ml_mode_convolutional)

    model_path = confm.get_universal_path(mconf_cremad_m.model_tflite_path)
    if not dfc.check_file_existence(model_path):
        print("Saved model does not exist for {database} - {gender}".format(database=confv.database_cremad, gender=confv.gender_male))
        sys.exit()

    # TFLite - Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TFLite - Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    y_prob = []
    count = 0
    for i in range(0, signal.shape[0] - mconf_cremad_m.step, mconf_cremad_m.step):
        count = count + 1  # Which 10th of a second number
        sample = signal[i:i + mconf_cremad_m.step]
        x = mfcc(sample, rate, numcep=mconf_cremad_m.nfeat, nfilt=mconf_cremad_m.nfilt, nfft=mconf_cremad_m.nfft)
        x = (x - mconf_cremad_m.min) / (mconf_cremad_m.max - mconf_cremad_m.min)

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
        y_pred.append(np.argmax(y_hat))

    fn_prob = np.mean(y_prob, axis=0).flatten()
    print("{database}-{gender}: The mean of all probabilities of {count} 1/10 second chunks from the single audio file for the classes {classes} are {probs}".format(database=mconf_cremad_m.database, gender=mconf_cremad_m.gender, count=count, classes=mconf_cremad_m.classes, probs=fn_prob))

    cremad_male_results_dict = calc.get_dict(mconf_cremad_m.classes, fn_prob)
    cremad_male_val_acc = calc.get_validation_accuracy(confm.get_universal_path(mconf_cremad_m.training_log_path))

    cremad_male_dict_act_map = confc.DictToValAccMap(dict=cremad_male_results_dict, val_acc=cremad_male_val_acc)

    return cremad_male_dict_act_map


def predict_shemo_male(signal='', rate=''):
    mconf_shemo_m = sl.load_model_config(database=confv.database_shemo, gender=confv.gender_male, mode=confv.ml_mode_convolutional)

    model_path = confm.get_universal_path(mconf_shemo_m.model_tflite_path)
    if not dfc.check_file_existence(model_path):
        print("Saved model does not exist for {database} - {gender}".format(database=confv.database_shemo, gender=confv.gender_male))
        sys.exit()

    # TFLite - Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TFLite - Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    y_prob = []
    count = 0
    for i in range(0, signal.shape[0] - mconf_shemo_m.step, mconf_shemo_m.step):
        count = count + 1  # Which 10th of a second number
        sample = signal[i:i + mconf_shemo_m.step]
        x = mfcc(sample, rate, numcep=mconf_shemo_m.nfeat, nfilt=mconf_shemo_m.nfilt, nfft=mconf_shemo_m.nfft)
        x = (x - mconf_shemo_m.min) / (mconf_shemo_m.max - mconf_shemo_m.min)

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
        y_pred.append(np.argmax(y_hat))

    fn_prob = np.mean(y_prob, axis=0).flatten()
    print("{database}-{gender}: The mean of all probabilities of {count} 1/10 second chunks from the single audio file for the classes {classes} are {probs}".format(database=mconf_shemo_m.database, gender=mconf_shemo_m.gender, count=count, classes=mconf_shemo_m.classes, probs=fn_prob))

    shemo_male_results_dict = calc.get_dict(mconf_shemo_m.classes, fn_prob)
    shemo_male_val_acc = calc.get_validation_accuracy(confm.get_universal_path(mconf_shemo_m.training_log_path))

    shemo_male_dict_act_map = confc.DictToValAccMap(dict=shemo_male_results_dict, val_acc=shemo_male_val_acc)

    return shemo_male_dict_act_map


def predict_ravdess_female(signal='', rate=''):
    mconf_ravdess_f = sl.load_model_config(database=confv.database_ravdess, gender=confv.gender_female, mode=confv.ml_mode_convolutional)

    model_path = confm.get_universal_path(mconf_ravdess_f.model_tflite_path)
    if not dfc.check_file_existence(model_path):
        print("Saved model does not exist for {database} - {gender}".format(database=confv.database_ravdess, gender=confv.gender_female))
        sys.exit()

    # TFLite - Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TFLite - Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    y_prob = []
    count = 0
    for i in range(0, signal.shape[0] - mconf_ravdess_f.step, mconf_ravdess_f.step):
        count = count + 1   # Which 10th of a second number
        sample = signal[i:i + mconf_ravdess_f.step]
        x = mfcc(sample, rate, numcep=mconf_ravdess_f.nfeat, nfilt=mconf_ravdess_f.nfilt, nfft=mconf_ravdess_f.nfft)
        x = (x - mconf_ravdess_f.min) / (mconf_ravdess_f.max - mconf_ravdess_f.min)

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
    print("{database}-{gender}: The mean of all probabilities of {count} 1/10 second chunks from the single audio file for the classes {classes} are {probs}".format(database=mconf_ravdess_f.database, gender=mconf_ravdess_f.gender, count=count, classes=mconf_ravdess_f.classes, probs=fn_prob))

    ravdess_female_results_dict = calc.get_dict(mconf_ravdess_f.classes, fn_prob)
    ravdess_female_val_acc = calc.get_validation_accuracy(confm.get_universal_path(mconf_ravdess_f.training_log_path))

    ravdess_female_dict_act_map = confc.DictToValAccMap(dict=ravdess_female_results_dict, val_acc=ravdess_female_val_acc)

    return ravdess_female_dict_act_map


def predict_emodb_female(signal='', rate=''):
    mconf_emodb_f = sl.load_model_config(database=confv.database_emodb, gender=confv.gender_female, mode=confv.ml_mode_convolutional)

    model_path = confm.get_universal_path(mconf_emodb_f.model_tflite_path)
    if not dfc.check_file_existence(model_path):
        print("Saved model does not exist for {database} - {gender}".format(database=confv.database_emodb, gender=confv.gender_female))
        sys.exit()

    # TFLite - Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TFLite - Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    y_prob = []
    count = 0
    for i in range(0, signal.shape[0] - mconf_emodb_f.step, mconf_emodb_f.step):
        count = count + 1  # Which 10th of a second number
        sample = signal[i:i + mconf_emodb_f.step]
        x = mfcc(sample, rate, numcep=mconf_emodb_f.nfeat, nfilt=mconf_emodb_f.nfilt, nfft=mconf_emodb_f.nfft)
        x = (x - mconf_emodb_f.min) / (mconf_emodb_f.max - mconf_emodb_f.min)

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
        y_pred.append(np.argmax(y_hat))

    fn_prob = np.mean(y_prob, axis=0).flatten()
    print("{database}-{gender}: The mean of all probabilities of {count} 1/10 second chunks from the single audio file for the classes {classes} are {probs}".format(database=mconf_emodb_f.database, gender=mconf_emodb_f.gender, count=count, classes=mconf_emodb_f.classes, probs=fn_prob))

    emodb_female_results_dict = calc.get_dict(mconf_emodb_f.classes, fn_prob)
    emodb_female_val_acc = calc.get_validation_accuracy(confm.get_universal_path(mconf_emodb_f.training_log_path))

    emodb_female_dict_act_map = confc.DictToValAccMap(dict=emodb_female_results_dict, val_acc=emodb_female_val_acc)

    return emodb_female_dict_act_map


def predict_cremad_female(signal='', rate=''):
    mconf_cremad_f = sl.load_model_config(database=confv.database_cremad, gender=confv.gender_female, mode=confv.ml_mode_convolutional)

    model_path = confm.get_universal_path(mconf_cremad_f.model_tflite_path)
    if not dfc.check_file_existence(model_path):
        print("Saved model does not exist for {database} - {gender}".format(database=confv.database_cremad, gender=confv.gender_female))
        sys.exit()

    # TFLite - Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TFLite - Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    y_prob = []
    count = 0
    for i in range(0, signal.shape[0] - mconf_cremad_f.step, mconf_cremad_f.step):
        count = count + 1  # Which 10th of a second number
        sample = signal[i:i + mconf_cremad_f.step]
        x = mfcc(sample, rate, numcep=mconf_cremad_f.nfeat, nfilt=mconf_cremad_f.nfilt, nfft=mconf_cremad_f.nfft)
        x = (x - mconf_cremad_f.min) / (mconf_cremad_f.max - mconf_cremad_f.min)

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
        y_pred.append(np.argmax(y_hat))

    fn_prob = np.mean(y_prob, axis=0).flatten()
    print("{database}-{gender}: The mean of all probabilities of {count} 1/10 second chunks from the single audio file for the classes {classes} are {probs}".format(database=mconf_cremad_f.database, gender=mconf_cremad_f.gender, count=count, classes=mconf_cremad_f.classes, probs=fn_prob))

    cremad_female_results_dict = calc.get_dict(mconf_cremad_f.classes, fn_prob)
    cremad_female_val_acc = calc.get_validation_accuracy(confm.get_universal_path(mconf_cremad_f.training_log_path))

    cremad_female_dict_act_map = confc.DictToValAccMap(dict=cremad_female_results_dict, val_acc=cremad_female_val_acc)

    return cremad_female_dict_act_map


def predict_shemo_female(signal='', rate=''):
    mconf_shemo_f = sl.load_model_config(database=confv.database_shemo, gender=confv.gender_female, mode=confv.ml_mode_convolutional)

    model_path = confm.get_universal_path(mconf_shemo_f.model_tflite_path)
    if not dfc.check_file_existence(model_path):
        print("Saved model does not exist for {database} - {gender}".format(database=confv.database_shemo, gender=confv.gender_female))
        sys.exit()

    # TFLite - Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TFLite - Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    y_prob = []
    count = 0
    for i in range(0, signal.shape[0] - mconf_shemo_f.step, mconf_shemo_f.step):
        count = count + 1  # Which 10th of a second number
        sample = signal[i:i + mconf_shemo_f.step]
        x = mfcc(sample, rate, numcep=mconf_shemo_f.nfeat, nfilt=mconf_shemo_f.nfilt, nfft=mconf_shemo_f.nfft)
        x = (x - mconf_shemo_f.min) / (mconf_shemo_f.max - mconf_shemo_f.min)

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
        y_pred.append(np.argmax(y_hat))

    fn_prob = np.mean(y_prob, axis=0).flatten()
    print("{database}-{gender}: The mean of all probabilities of {count} 1/10 second chunks from the single audio file for the classes {classes} are {probs}".format(database=mconf_shemo_f.database, gender=mconf_shemo_f.gender, count=count, classes=mconf_shemo_f.classes, probs=fn_prob))

    shemo_female_results_dict = calc.get_dict(mconf_shemo_f.classes, fn_prob)
    shemo_female_val_acc = calc.get_validation_accuracy(confm.get_universal_path(mconf_shemo_f.training_log_path))

    shemo_female_dict_act_map = confc.DictToValAccMap(dict=shemo_female_results_dict, val_acc=shemo_female_val_acc)

    return shemo_female_dict_act_map
'''