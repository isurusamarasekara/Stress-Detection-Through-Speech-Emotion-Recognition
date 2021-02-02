import pandas as pd
import backbone.support.configurations_variables as confv
import os
from tqdm import tqdm
import librosa


def load_original_data(database):
    data_info_df = pd.DataFrame()

    if database == confv.database_ravdess:
        data_info_df = load_data_intel_ravdess_original()
    elif database == confv.database_emodb:
        data_info_df = load_data_intel_emodb_original()
    elif database == confv.database_cremad:
        data_info_df = load_data_intel_cremad_original()
    elif database == confv.database_shemo:
        data_info_df = load_data_intel_shemo_original()

    return data_info_df


def load_data_intel_ravdess_original():
    data_info_df = pd.DataFrame(columns=['audio_fname', 'actor_no', 'gender', 'emotion_no', 'primary_emotion'])
    count = 0

    actr_dir_list = os.listdir(confv.smpl_data_path_ravdess)
    for act_dir in tqdm(actr_dir_list):
        actr_dir_path = os.path.join(confv.smpl_data_path_ravdess, act_dir)
        aud_fl_list = os.listdir(actr_dir_path)
        for aud_fl in aud_fl_list:
            ind_aud_fl_ids = aud_fl.split('.')[0].split('-')
            # aud_fl_pth = os.path.join(actr_dir_path, aud_fl)
            audio_fname = aud_fl
            actor_no = ind_aud_fl_ids[-1]

            if int(actor_no) % 2 == 0:
                gender = confv.gender_female
            else:
                gender = confv.gender_male

            emotion_no = ind_aud_fl_ids[2]
            primary_emotion = assign_primary_emotion(int(emotion_no), database=confv.database_ravdess)

            data_info_df.loc[count] = [audio_fname, actor_no, gender, emotion_no, primary_emotion]
            count += 1

    return data_info_df


def load_data_intel_emodb_original():
    data_info_df = pd.DataFrame(columns=['audio_fname', 'speaker_no', 'gender', 'emotion_ltr', 'primary_emotion'])
    count = 0

    aud_fl_list = os.listdir(confv.smpl_data_path_emodb)
    for aud_fl in tqdm(aud_fl_list):
        pure_name = aud_fl.split('.')[0]
        audio_fname = aud_fl
        speaker_no = pure_name[:2]

        if speaker_no in {"03", "10", "11", "12", "15"}:
            gender = confv.gender_male
        elif speaker_no in {"08", "09", "13", "14", "16"}:
            gender = confv.gender_female
        else:
            gender = confv.gender_nonconformity

        emotion_ltr = pure_name[5]
        primary_emotion = assign_primary_emotion(emotion_ltr, database=confv.database_emodb)

        data_info_df.loc[count] = [audio_fname, speaker_no, gender, emotion_ltr, primary_emotion]
        count += 1

    return data_info_df


def load_data_intel_cremad_original():
    data_info_df = pd.DataFrame(columns=['audio_fname', 'actor_id', 'gender', 'emotion_abbr', 'primary_emotion'])
    count = 0

    aud_fl_list = os.listdir(confv.smpl_data_path_cremad)
    for aud_fl in tqdm(aud_fl_list):
        ind_aud_fl_ids = aud_fl.split('.')[0].split('_')
        audio_fname = aud_fl
        actor_id = ind_aud_fl_ids[0]
        gender = get_cremad_gender(actor_id)
        emotion_abbr = ind_aud_fl_ids[2]
        primary_emotion = assign_primary_emotion(emotion_abbr, database=confv.database_cremad)


        data_info_df.loc[count] = [audio_fname, actor_id, gender, emotion_abbr, primary_emotion]
        count += 1

    return data_info_df


def load_data_intel_shemo_original():
    data_info_df = pd.DataFrame(columns=['audio_fname', 'speaker_no', 'gender', 'emotion_ltr', 'primary_emotion'])
    count = 0

    gender_dir_list = os.listdir(confv.smpl_data_path_shemo)
    for gender_dir in tqdm(gender_dir_list):
        gender_dir_path = os.path.join(confv.smpl_data_path_shemo, gender_dir)
        aud_fl_list = os.listdir(gender_dir_path)
        for aud_fl in aud_fl_list:
            audio_fname = aud_fl
            speaker_no = audio_fname[1:3]

            if audio_fname[0] == 'F':
                gender = confv.gender_female
            elif audio_fname[0]:
                gender = confv.gender_male
            else:
                gender = confv.gender_nonconformity

            emotion_ltr = audio_fname[3]
            primary_emotion = assign_primary_emotion(emotion_ltr, database=confv.database_shemo)

            data_info_df.loc[count] = [audio_fname, speaker_no, gender, emotion_ltr, primary_emotion]
            count += 1

    return data_info_df


def assign_primary_emotion(input, database):
    output = ""
    if database == confv.database_ravdess:
        output = assign_primary_emotion_ravdess(input)
    elif database == confv.database_emodb:
        output = assign_primary_emotion_emodb(input)
    elif database == confv.database_cremad:
        output = assign_primary_emotion_cremad(input)
    elif database == confv.database_shemo:
        output = assign_primary_emotion_shemo(input)

    return output


def assign_primary_emotion_ravdess(input):
    output = ""
    if input == 1:
        output = confv.N
    elif input == 2:
        output = confv.C
    elif input == 3:
        output = confv.H
    elif input == 4:
        output = confv.S
    elif input == 5:
        output = confv.A
    elif input == 6:
        output = confv.F
    elif input == 7:
        output = confv.D
    elif input == 8:
        output = confv.Su
    else:
        output = confv.Non

    return output


def assign_primary_emotion_emodb(input):
    output = ""
    if input == "W":
        output = confv.A
    elif input == "L":
        output = confv.B
    elif input == "E":
        output = confv.D
    elif input == "A":
        output = confv.F
    elif input == "F":
        output = confv.H
    elif input == "T":
        output = confv.S
    elif input == "N":
        output = confv.N
    else:
        output = confv.Non

    return output


def assign_primary_emotion_cremad(input):
    output = ""
    if input == "ANG":
        output = confv.A
    elif input == "DIS":
        output = confv.D
    elif input == "FEA":
        output = confv.F
    elif input == "HAP":
        output = confv.H
    elif input == "NEU":
        output = confv.N
    elif input == "SAD":
        output = confv.S
    else:
        output = confv.Non

    return output


def assign_primary_emotion_shemo(input):
    output = ""
    if input == "S":
        output = confv.S
    elif input == "A":
        output = confv.A
    elif input == "H":
        output = confv.H
    elif input == "W":
        output = confv.Su
    elif input == "F":
        output = confv.F
    elif input == "N":
        output = confv.N
    else:
        output = confv.Non

    return output


def get_cremad_gender(actor_id):
    id_info_cremad_df = pd.read_csv(confv.info_path_cremad)
    csv_gender = id_info_cremad_df.loc[id_info_cremad_df['ActorID'] == int(actor_id), 'Sex'].iloc[0]

    gender = ""
    if csv_gender.lower() == confv.gender_female:
        gender = confv.gender_female
    elif csv_gender.lower() == confv.gender_male:
        gender = confv.gender_male
    else:
        gender = confv.gender_nonconformity

    return gender


def get_audio_file_path(audio_fname, database, status, gender=confv.gender_nonconformity):
    aud_fl_path = ""
    if status == confv.original:
        aud_fl_path = get_original_audio_file_path(audio_fname=audio_fname, database=database)
    elif status == confv.clean:
        aud_fl_path = get_clean_audio_file_path(database=database, gender=gender, audio_fname=audio_fname)

    return aud_fl_path


def get_original_audio_file_path(audio_fname, database):
    aud_fl_path = ""
    if database == confv.database_ravdess:
        actor_no = audio_fname.split('.')[0].split('-')[-1]
        actor_dir = "Actor_" + actor_no.zfill(2)
        # actor_dir = "Actor_" + '%0.2d' % actor_no
        aud_fl_path = os.path.join(confv.smpl_data_path_ravdess, actor_dir, audio_fname)

    elif database == confv.database_emodb:
        aud_fl_path = os.path.join(confv.smpl_data_path_emodb, audio_fname)

    elif database == confv.database_cremad:
        aud_fl_path = os.path.join(confv.smpl_data_path_cremad, audio_fname)

    elif database == confv.database_shemo:
        gender_dir = ""
        if audio_fname[0] == 'F':
            gender_dir = confv.gender_female
        elif audio_fname[0] == 'M':
            gender_dir = confv.gender_male
        else:
            pass
        aud_fl_path = os.path.join(confv.smpl_data_path_shemo, gender_dir, audio_fname)
    return aud_fl_path


def get_clean_audio_file_path(database, gender, audio_fname):
    aud_fl_path = ""
    if database == confv.database_ravdess:
        if gender == confv.gender_male:
            aud_fl_path = os.path.join(confv.base_store, confv.clean_audio, confv.dataset_ravdess_male, audio_fname)
        elif gender == confv.gender_female:
            aud_fl_path = os.path.join(confv.base_store, confv.clean_audio, confv.dataset_ravdess_female, audio_fname)
    elif database == confv.database_emodb:
        if gender == confv.gender_male:
            aud_fl_path = os.path.join(confv.base_store, confv.clean_audio, confv.dataset_emodb_male, audio_fname)
        elif gender == confv.gender_female:
            aud_fl_path = os.path.join(confv.base_store, confv.clean_audio, confv.dataset_emodb_female, audio_fname)
    elif database == confv.database_cremad:
        if gender == confv.gender_male:
            aud_fl_path = os.path.join(confv.base_store, confv.clean_audio, confv.dataset_cremad_male, audio_fname)
        elif gender == confv.gender_female:
            aud_fl_path = os.path.join(confv.base_store, confv.clean_audio, confv.dataset_cremad_female, audio_fname)
    elif database == confv.database_shemo:
        if gender == confv.gender_male:
            aud_fl_path = os.path.join(confv.base_store, confv.clean_audio, confv.dataset_shemo_male, audio_fname)
        elif gender == confv.gender_female:
            aud_fl_path = os.path.join(confv.base_store, confv.clean_audio, confv.dataset_shemo_female, audio_fname)
    return aud_fl_path


def get_df_with_length(df, database, status, gender):
    df1 = df.copy()
    '''
        for index in range(len(data_info_df)):
            rate, signal = wavfile.read(data_info_df.audio_file_path[index])
            data_info_df.


        '''

    df1.set_index('audio_fname', inplace=True)

    # rate, signal = wavfile.read(data_info_df.audio_file_path[400]) # 0, 1, 500
    # print(rate)
    # rate, signal = wavfile.read(data_info_df.audio_file_path[401])
    # print(rate)

    for audio_fname in tqdm(df1.index):
        audio_file_path = get_audio_file_path(audio_fname, database=database, status=status, gender=gender)
        signal, rate = librosa.load(audio_file_path, sr=None)
        # print(rate)
        df1.at[audio_fname, 'length'] = signal.shape[0] / rate

    df1.reset_index(inplace=True)

    return df1
