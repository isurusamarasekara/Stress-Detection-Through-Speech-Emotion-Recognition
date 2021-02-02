import pandas as pd
import numpy as np
from speech_analysis_raspi.support import configurations_variables as confv
from speech_analysis_raspi.support import recording_configurations as rconf


def apply_envelope(signal, threshold, application_type):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    # This window is not bound to build features in model training. But better use the step_size there.
    # So, imitation bound.
    if application_type == confv.app_type_prerecorded_upload or application_type == confv.app_type_prerecorded_time_specified:
        # window=confv.step_size is used due to the use of confv.resample_rate at the time of loading audio for these to types,
        # and that the dependent of confv.resaple_rate for nth of a second is confv.step_size.
        y_mean = y.rolling(window=confv.step_size, min_periods=1, center=True).mean()
    elif application_type == confv.app_type_prerecorded_time_specified_noise or application_type == confv.app_type_real_time:
        # I can use confv.step_size here too. But just... ?
        y_mean = y.rolling(window=int(rconf.CHUNKSIZE_realtime), min_periods=1, center=True).mean()

    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    signal = signal[mask]

    return signal


def get_dict(classes, probs):
    dict = {}
    for i in range(len(classes)):
        dict[classes[i]] = probs[i]

    return dict


def get_validation_accuracy(training_log_path):
    df = pd.read_csv(training_log_path)
    val_acc = df['val_accuracy'].iloc[-1]

    return val_acc


def get_biased_probs(dict_act_map_list: list):
    stressed = confv.class_stressed
    not_stressed = confv.class_not_stressed

    stressed_upper = 0
    not_stressed_upper = 0
    down = 0
    for item in dict_act_map_list:
        stressed_upper = stressed_upper + (item.dict[stressed] * item.val_acc)
        # print('stressed_upper', stressed_upper)
        # print('item.dict[stressed]', item.dict[stressed])
        # print('item.val_acc', item.val_acc)
        not_stressed_upper = not_stressed_upper + (item.dict[not_stressed] * item.val_acc)
        # print('not_stressed_upper', not_stressed_upper)
        # print('item.dict[not_stressed]', item.dict[not_stressed])
        # print('item.val_acc', item.val_acc)
        down = down + item.val_acc
        # print('down', down)

    stressed_final_prob = round((stressed_upper / down), 4)
    # print('stressed_final_prob', stressed_final_prob)
    not_stressed_final_prob = round((not_stressed_upper / down), 4)
    # print('not_stressed_final_prob', not_stressed_final_prob)

    final_result = {stressed: stressed_final_prob, not_stressed: not_stressed_final_prob}

    return final_result
