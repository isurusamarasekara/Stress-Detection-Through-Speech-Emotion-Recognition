import os
import backbone.support.directory_file_checking as dfc
import backbone.support.configurations_variables as confv
import pickle
import backbone.support.custom_exceptions as ce
import sys


def save_dataframe(dataframeobj):
    if not os.path.isfile(dataframeobj.save_path):
        dfc.check_dir(confv.saved_dataframes)
        with open(dataframeobj.save_path, 'wb') as outfile:
            pickle.dump(dataframeobj, outfile, protocol=2)

        print("\tDataframe successfully saved to: {relative_path}".format(relative_path=dataframeobj.save_path))

    else:
        print("\tDataframe already found in: {relative_path}".format(relative_path=dataframeobj.save_path))


def load_dataframe(dataframeobj):
    try:
        if os.path.isfile(dataframeobj.save_path):
            with open(dataframeobj.save_path, 'rb') as infile:
                dataframeobj = pickle.load(infile)

        else:
            raise ce.DataframeLoadingError

    except ce.DataframeLoadingError:
        print("\tDataframe NOT found for: ")
        print("\t\tDatabase: {database}".format(database=dataframeobj.database))
        print("\t\tGender: {gender}".format(gender=dataframeobj.gender))
        print("\t\tDataset Name: {dataset}".format(dataset=dataframeobj.dataset))
        print("\tFirst save the dataframe in path \"{relative_path}\" and run the program again.".format(relative_path=dataframeobj.save_path))
        print("\n\nEXITING THE PROGRAM")
        sys.exit()

    return dataframeobj
