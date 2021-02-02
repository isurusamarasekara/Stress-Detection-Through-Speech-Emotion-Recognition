import backbone.support.configurations_variables as confv


def assign_stress_emotion_class_2(df):
    df1 = df.copy()
    all_stress_emotions_list = []

    for index in range(len(df1)):
        if df1.primary_emotion[index] in {confv.N, confv.C, confv.H, confv.Su, confv.B}:
            secondary_emotion = "Not Stressed"
        elif df1.primary_emotion[index] in {confv.S, confv.A, confv.F, confv.D}:
            secondary_emotion = "Stressed"
        else:
            secondary_emotion = "none"

        all_stress_emotions_list.append(secondary_emotion)

    df1['stress_emotion'] = all_stress_emotions_list

    return df1


def remove_none_emotions(df):
    df1 = df.copy()

    df1 = df1[df1.primary_emotion != confv.Non]
    df1 = df1[df1.stress_emotion != confv.Non]
    df1.reset_index(drop=True, inplace=True)

    return df1


def isolate_by_gender(df, gender):
    df1 = df.copy()
    if gender == confv.gender_female:
        df1 = isolate_by_female(df1)
    elif gender == confv.gender_male:
        df1 = isolate_by_male(df1)

    return df1


def isolate_by_female(df):
    df1 = df.copy()
    df1 = df1[df1.gender != confv.gender_male]
    df1.reset_index(drop=True, inplace=True)

    return df1


def isolate_by_male(df):
    df1 = df.copy()
    df1 = df1[df1.gender != confv.gender_female]
    df1.reset_index(drop=True, inplace=True)

    return df1
