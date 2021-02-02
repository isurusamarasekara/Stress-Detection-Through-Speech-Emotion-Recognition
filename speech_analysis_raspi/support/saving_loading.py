import os
from speech_analysis_raspi.support import configurations_variables as confv
from speech_analysis_raspi.support import directory_file_checking as dfc
from speech_analysis_raspi.support import remapping_modules as rm
import sys


def load_model_config(database, gender, mode):
    # print("\t------Model config information loading started for the {database} - {gender}------".format(database=database, gender=gender))

    model_config_path = os.path.join(confv.base_store, confv.saved_modelconfigs, database, gender, mode)
    # print(model_config_path)
    if dfc.check_file_existence(model_config_path):
        with open(model_config_path, 'rb') as infile:
            modelconfig = rm.modified_load(infile)

        # print("\t------Model config information loading finished for the {database} - {gender}------".format(database=database, gender=gender))

    else:
        print("ModelConfig file does not exist for {database} - {gender}".format(database=database, gender=gender))
        sys.exit()

    return modelconfig
