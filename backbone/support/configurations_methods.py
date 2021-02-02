import backbone.support.configurations_variables as confv


def get_evelope_threshold(database, gender):
    envelope_threshold = 0.0005
    if database == confv.database_ravdess:
        if gender == confv.gender_male:
            envelope_threshold = 0.003
        elif gender == confv.gender_female:
            envelope_threshold = 0.0025
    elif database == confv.database_emodb:
        if gender == confv.gender_male:
            envelope_threshold = 0.09
        elif gender == confv.gender_female:#0.05
            envelope_threshold = 0.06   # Don't get it up to 0.5
    elif database == confv.database_cremad:     # Cremad (Shemo also) could improve to 0.05
        if gender == confv.gender_male:
            envelope_threshold = 0.05
        elif gender == confv.gender_female:
            envelope_threshold = 0.02
    elif database == confv.database_shemo:
        if gender == confv.gender_male:
            envelope_threshold = 0.03
        elif gender == confv.gender_female:
            envelope_threshold = 0.02

    return envelope_threshold
