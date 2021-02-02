import backbone.support.classes_and_adjustments as ca
import backbone.support.configurations_variables as confv
import numpy as np
import backbone.support.directory_file_checking as dfc
import os
from tqdm import tqdm
import backbone.support.data_loading as dl
import librosa
import backbone.support.calculations as calc
import backbone.support.configurations_methods as confm
from scipy.io import wavfile
import backbone.support.custom_exceptions as ce
import sys


def data_adjustments(data_info_df):
    # Stressed/ Not Stressed Class Assignment
    print("\t\t-----Assigning Stress/ Not Stressed-----")
    data_info_df = ca.assign_stress_emotion_class_2(data_info_df)
    print("\tNumber of sample audio files afterwards: ", len(data_info_df))
    # print(data_info_df.head())
    # print(data_info_df.tail())

    # Remove "none" emotion
    print("\t\t-----Removing None Emotion-----")
    data_info_df = ca.remove_none_emotions(data_info_df)
    print("\tNumber of sample audio files afterwards: ", len(data_info_df))
    # print(data_info_df.head())
    # print(data_info_df.tail())

    # Gender Isolation
    ## Male gender isolation
    data_info_df_M = ca.isolate_by_gender(data_info_df, gender=confv.gender_male)
    print("\t\t-----Isolating Male Data-----")
    print("\tNumber of sample audio files afterwards: ", len(data_info_df_M))
    # print(data_info_df_M.head())
    # print(data_info_df_M.tail())

    ## Female gender isolation
    data_info_df_F = ca.isolate_by_gender(data_info_df, gender=confv.gender_female)
    print("\t\t-----Isolating Female Data-----")
    print("\tNumber of sample audio files afterwards: ", len(data_info_df_F))
    # print(data_info_df_F.head())
    # print(data_info_df_F.tail())

    return data_info_df_M, data_info_df_F


def data_cleaning(df, database):
    df1 = df.copy()
    gender_list = list(np.unique(df1.gender))
    if len(gender_list) == 1:
        gender = gender_list[0]
    else:
        pass  # Exception

    # clean_dir = os.path.join(os.getcwd(), 'clean1')
    if database == confv.database_ravdess and gender == confv.gender_male:
        base_clean_dir = confv.dataset_ravdess_male
    elif database == confv.database_ravdess and gender == confv.gender_female:
        base_clean_dir = confv.dataset_ravdess_female
    elif database == confv.database_emodb and gender == confv.gender_male:
        base_clean_dir = confv.dataset_emodb_male
    elif database == confv.database_emodb and gender == confv.gender_female:
        base_clean_dir = confv.dataset_emodb_female
    elif database == confv.database_cremad and gender == confv.gender_male:
        base_clean_dir = confv.dataset_cremad_male
    elif database == confv.database_cremad and gender == confv.gender_female:
        base_clean_dir = confv.dataset_cremad_female
    elif database == confv.database_shemo and gender == confv.gender_male:
        base_clean_dir = confv.dataset_shemo_male
    elif database == confv.database_shemo and gender == confv.gender_female:
        base_clean_dir = confv.dataset_shemo_female

    clean_dir = dfc.check_dir(base_clean_dir)
    try:
        if len(os.listdir(clean_dir)) == 0:
            for audio_fname in tqdm(df1.audio_fname):
                aud_fl_pth = dl.get_audio_file_path(audio_fname, database=database, status=confv.original, gender=gender)
                # print(df1.audio_fname[df1.audio_file_path == aud_fl_pth])
                signal, rate = librosa.load(aud_fl_pth, sr=confv.resample_rate)
                envelope_threshold = confm.get_evelope_threshold(database=database, gender=gender)
                mask = calc.envelope(signal=signal, threshold=envelope_threshold)
                # print('clean/' + df1[df1.audio_file_path == aud_fl_pth].iloc[0, 0])

                full_pth_nm = dl.get_audio_file_path(database=database, status=confv.clean, gender=gender, audio_fname=audio_fname)
                # full_pth_nm = os.path.join(clean_dir + audio_fname)
                wavfile.write(filename=full_pth_nm, rate=rate, data=signal[mask])

            print("\tData successfully cleaned for: ")
            print("\t\tDatabase: {database}".format(database=database))
            print("\t\tGender: {gender}".format(gender=gender))
            print("\t\tDataset Name: {name}".format(name=base_clean_dir))
            print("\tstored to the path: {relative_path}".format(relative_path=clean_dir))

        elif len(os.listdir(clean_dir)) < len(df1): # What if >
            raise ce.DataCleaningError

        elif len(os.listdir(clean_dir)) == len(df1):
            print("\tCleaned data already found for: ")
            print("\t\tDatabase: {database}".format(database=database))
            print("\t\tGender: {gender}".format(gender=gender))
            print("\t\tDataset Name: {name}".format(name=base_clean_dir))
            print("\tat the path: {relative_path}".format(relative_path=clean_dir))

    except ce.DataCleaningError:
        print("\tPartially cleaned data found for: ")
        print("\t\tDatabase: {database}".format(database=database))
        print("\t\tGender: {gender}".format(gender=gender))
        print("\t\tDataset Name: {name}".format(name=base_clean_dir))
        print("\tFirst either delete the folder in path \"{relative_path}\", or delete the files residing in it".format(relative_path=clean_dir))
        print("\tand run the program again")

        print("\nEXITING THE PROGRAM")
        sys.exit()


def check_and_adjust_df_for_minimum_audio_length_after_cleaning(df, database, gender):
    df1 = df.copy()
    df1 = dl.get_df_with_length(df=df1, database=database, status=confv.clean, gender=gender)

    condition = confv.step_size / confv.resample_rate
    df1 = df1[df1.length > condition]
    df1.reset_index(drop=True, inplace=True)

    if len(df) > len(df1):
        print("\n*/*/*/*/*/*/*/*/*/*/*/----Dataframe was updated to adhere to the minimum audio length requirement----/*/*/*/*/*/*/*/*/*/*/*")
        print(len(df1))
        print(df1.head())
        print(df1.tail())
        df1.drop('length', axis=1, inplace=True)
    else:
        print("*/*/*/*/*/*/*/*/*/*/*/----Dataframe already adheres to the minimum audio length requirement----/*/*/*/*/*/*/*/*/*/*/*")
        df1 = df

    return df1
