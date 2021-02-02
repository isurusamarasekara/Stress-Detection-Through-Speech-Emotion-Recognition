import backbone.support.directory_file_checking as dfc
import pickle
from tqdm import tqdm
import numpy as np
import backbone.support.data_loading as dl
import backbone.support.configurations_variables as confv
import librosa
from python_speech_features import mfcc
# from keras.utils.np_utils import to_categorical
import backbone.support.configuration_classes as confc


def check_data(modelconfig): # Check for saved_modelconfigs and whether saved features are less than calculated
    if dfc.check_file_existence(modelconfig.feature_path):
        print('\t\t\tExisting built features found for:\n'
              '\t\t\t\tDatabase: {database}\n'
              '\t\t\t\tGender: {gender}\n'
              '\t\t\t\tML Model: {model}\n'
              '\t\t\tLoading these existing data features.'.format(database=modelconfig.database, gender=modelconfig.gender, model=modelconfig.mode))
        with open(modelconfig.feature_path, 'rb') as infile:
            feature_set = pickle.load(infile)
            return feature_set
    else:
        return None


def build_random_features(modelconfig, randfeatparams):
    print("\t-----Feature building started for the {database} - {gender} - {mode} model-----".format(database=modelconfig.database, gender=modelconfig.gender, mode=modelconfig.mode))
    feature_set = check_data(modelconfig)
    if feature_set:
        return feature_set.X, feature_set.y

    df = randfeatparams.df1
    n_samples = randfeatparams.n_samples
    class_dist = randfeatparams.class_dist
    prob_dist = randfeatparams.prob_dist
    classes = randfeatparams.classes
    print(modelconfig.classes)
    print(randfeatparams.classes)

    a = range(n_samples)[-1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        if _ == a:
            print("rand_class: ", rand_class)
        audio_file = np.random.choice(df[df.stress_emotion == rand_class].audio_fname)
        # print(audio_file)
        aud_fl_pth = dl.get_audio_file_path(audio_fname=audio_file, database=modelconfig.database, status=confv.clean, gender=modelconfig.gender)
        signal, rate = librosa.load(aud_fl_pth, sr=None)
        if _ == a:
            print("signal---------------------", signal.shape)
        # print(rand_class)
        # print(df.loc[df['audio_fname'] == audio_file, 'stress_emotion'].iloc[0])
        stress_emotion = df.loc[df['audio_fname'] == audio_file, 'stress_emotion'].iloc[0]
        if _ == a:
            print("Stress Emotion: ", stress_emotion)
            print("Stress Emotion Index: ", classes.index(stress_emotion))
        # print(aud_fl_pth)
        # print(signal.shape[0])
        # print(rate)
        # print(signal.shape[0]/rate)
        # print(modelconfig.step)
        # print(signal.shape[0]/modelconfig.step)
        # print(signal.shape[0] - modelconfig.step)
        rand_index = np.random.randint(0, signal.shape[0] - modelconfig.step)
        sample = signal[rand_index:rand_index + modelconfig.step]
        if _ == a:
            print("sample---------------------", sample.shape)
        X_sample = mfcc(sample, rate, numcep=modelconfig.nfeat, nfilt=modelconfig.nfilt, nfft=modelconfig.nfft)
        if _ == a:
            print("X_sample---------------------", X_sample.shape)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        if _ == a:
            print("X---------------------", len(X))
        y.append(classes.index(stress_emotion))

    modelconfig.min = _min
    modelconfig.max = _max
    X, y = np.array(X), np.array(y)
    print("@@@@@@@@   y-- ", y)
    print("@@@@@@@    ", X.shape)
    X = (X - _min) / (_max - _min)
    print("#########   ", X.shape)
    print("#########   ", y.shape)
    if modelconfig.mode == confv.ml_mode_convolutional:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)    #  X [no_samples, 13 ,9 ,1]
        print("^^^^^^^^^^      ", X.shape)
        print("^^^^^^^^^^ Should be similar to no of samples     ", X.shape[0])
    elif modelconfig.mode == 'a':
        pass
    # p = y
    # y = to_categorical(y, num_classes=len(classes))
    # print(y)
    # p = np.eye(len(classes))[p]
    # print(p)
    # print(y == p)
    # return

    y = np.eye(len(classes))[y]

    # modelconfig.set_feature_set(X, y)  # python allows you to add new fields to the objects on the fly.
    feature_set = confc.FeatureSet(X, y)

    dfc.check_dir_inside_saved_features_and_modelconfigs_and_models(parent=confv.saved_features, database=modelconfig.database, gender=modelconfig.gender)
    print('\t\t\t-----Writing data features-----')
    with open(modelconfig.feature_path, 'wb') as outfile:
        pickle.dump(feature_set, outfile, protocol=2)

    dfc.check_dir_inside_saved_features_and_modelconfigs_and_models(parent=confv.saved_modelconfigs, database=modelconfig.database, gender=modelconfig.gender)
    print('\t\t\t-----Writing model configuration information-----')
    with open(modelconfig.model_config_path, 'wb') as outfile:
        pickle.dump(modelconfig, outfile, protocol=2)

    print("\t-----Feature building finished for the {database} - {gender} - {mode} model-----".format(database=modelconfig.database, gender=modelconfig.gender, mode=modelconfig.mode))
    return X, y
