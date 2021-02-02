from backbone_independent.support import directory_file_checking as dfc
from backbone_independent.support import configurations_variables as confv
import os
import simpleaudio as sa
import sys
import librosa
from backbone_independent.support import predict as pr


def play(audio_file_path):
    if os.path.isfile(audio_file_path):
        print('Started playing recorded audio file')
        wave_obj = sa.WaveObject.from_wave_file(audio_file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        print('Finished playing recorded audio file')

    else:
        print('No file found at', audio_file_path)
        sys.exit()

    return


def main():
    # Create the upload directory if not already created
    dfc.check_dir(directory=confv.prerecorded_upload_dir)

    # Clean and empty the uploads directory, just in case, before any upload takes place
    # dfc.clean_relevant_directory(confv.prerecorded_upload_dir)

    # User selects the gender
    gender = confv.gender_male

    # User uploads the file

    # Check if the directory is empty
    isEmpty = dfc.is_directory_empty(confv.prerecorded_upload_dir)

    if not isEmpty:
        # File is selected by the program
        aud_fl_pth = dfc.get_audio_file_path_from_directory(confv.prerecorded_upload_dir)

        # Playing the audio
        play(audio_file_path=aud_fl_pth)

        # Load signal - resampling
        signal, rate = librosa.load(path=aud_fl_pth, sr=confv.resample_rate)

        # Check for minimum length compliance - apply envelope
        isComplied, signal = dfc.check_for_minimum_length_compliance(signal=signal, threshold=confv.prerecorded_upload_envelope_threshold, application_type=confv.app_type_prerecorded_upload)
        if isComplied:
            # Predict the stress emotion
            final_result_dict = pr.predict_upload(signal=signal, gender=gender)
            print("The stress prediction: ", final_result_dict[confv.class_stressed])
            print(final_result_dict)

            None

        else:
            None
            print("Not complied - main")

        # Clean and empty the prerecorded-uploads directory
        # dfc.clean_relevant_directory(confv.prerecorded_upload_dir)

    else:
        None
        print("Empty - main")


main()
