import pyaudio
from speech_analysis_raspi.support import directory_file_checking as dfc
from speech_analysis_raspi.support import configurations_variables as confv
from speech_analysis_raspi.support import recording_configurations as rconf
import os
from speech_analysis_raspi.support import prerecorded_predict as pr
import sounddevice as sd
from speech_analysis_raspi.support import plotting as pl
import soundfile as sf
import numpy as np
import simpleaudio as sa
import sys
import librosa


def record(time_to_record, audio_save_path):
    print('Started recording audio using sounddevice... For {time} seconds'.format(time=time_to_record))
    audio_signal = sd.rec(frames=int(time_to_record * rconf.RATE_prerecorded_time_specified), samplerate=rconf.RATE_prerecorded_time_specified, channels=rconf.CHANNELS, dtype='float32')
    sd.wait()
    print('Finished recording audio.')

    # pl.plot_audio(signal=audio_signal, loudness_threshold=confv.prerecorded_time_specified_envelope_threshold)

    print('Saving audio file to', audio_save_path)
    sf.write(file=audio_save_path, data=audio_signal, samplerate=rconf.RATE_prerecorded_time_specified)

    return

    # Create an interface to PortAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=rconf.FORMAT, channels=rconf.CHANNELS, rate=rconf.RATE_prerecorded_time_specified, frames_per_buffer=rconf.CHUNKSIZE_prerecorded_time_specified, input=True, input_device_index=1)

    print('Started recording audio using pyaudio and np.float convert... For {time} seconds'.format(time=time_to_record))
    # Initialize array to store frames
    audio_signal = []
    # Capture raw byte data in chunks, converting them to np.float32 and store using soundfile.
    for i in range(0, int(rconf.RATE_prerecorded_time_specified/rconf.CHUNKSIZE_prerecorded_time_specified * time_to_record)):
        raw_data_buffer = stream.read(num_frames=rconf.CHUNKSIZE_prerecorded_time_specified)     # The number of frames to read.
        # frames.append(raw_data)
        readable_data_buffer = np.frombuffer(raw_data_buffer, dtype=rconf.NP_DATATYPE)

        if i == 0:
            audio_signal = readable_data_buffer
        else:
            audio_signal = np.concatenate((audio_signal, readable_data_buffer), axis=None)

    # pl.plot_audio(signal=audio_signal, loudness_threshold=confv.prerecorded_time_specified_envelope_threshold)

    # Stop stream
    stream.stop_stream()
    stream.close()
    # Close the PortAudio interface
    p.terminate()
    print('Finished recording audio.')

    print('Saving audio file to', audio_save_path)
    sf.write(file=audio_save_path, data=audio_signal, samplerate=rconf.RATE_prerecorded_time_specified)

    return


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
    # Create the realtime directory if not already created
    dfc.check_dir(directory=confv.prerecorded_time_specified_dir)

    # Clean and empty the uploads directory, just in case, before any upload takes place
    dfc.clean_relevant_directory(directory=confv.prerecorded_time_specified_dir)

    # User selects the gender
    gender = confv.gender_male

    # User input number of seconds to record
    prerecorded_time_specified_seconds = 10

    aud_fl_pth = confv.prerecorded_time_specified_file_aud_fl_pth     # If multiple users can use, then the audio path needs to be joined here rather than getting from the confv
    record(time_to_record=prerecorded_time_specified_seconds, audio_save_path=aud_fl_pth)

    # Check if the directory is empty
    isEmpty = dfc.is_directory_empty(directory=confv.prerecorded_time_specified_dir)
    if not isEmpty:
        # File is set to be selected by the program from the directory just in case it is saved by another name than
        # aud_fl_pth - which should be made impossible.
        # If multiple users can use, the file path should come from above
        aud_fl_pth = dfc.get_audio_file_path_from_directory(directory=confv.prerecorded_time_specified_dir)

        # Playing the audio
        play(audio_file_path=aud_fl_pth)

        # Load signal - resampling
        signal, rate = librosa.load(path=aud_fl_pth, sr=confv.resample_rate)    # can't use sf.read due to resampling at the load time
        # pl.plot_audio(signal=signal, loudness_threshold=confv.prerecorded_time_specified_envelope_threshold)    # Make the threshold confv.prerecorded_time_specified_envelope_threshold_raspi_issue if running on raspberry pi with touching a metal section

        # Check for minimum length compliance - apply envelope
        # Make the threshold confv.prerecorded_time_specified_envelope_threshold_raspi_issue if running on raspberry pi with touching a metal section
        isComplied, signal = dfc.check_for_minimum_length_compliance(signal=signal, threshold=confv.prerecorded_time_specified_envelope_threshold, application_type=confv.app_type_prerecorded_time_specified)
        if isComplied:
            # If not complied, having this plot outside of the conditional statement will have a blank plot
            # pl.plot_audio(signal=signal, loudness_threshold=confv.prerecorded_time_specified_envelope_threshold)    # Make the threshold confv.prerecorded_time_specified_envelope_threshold_raspi_issue if running on raspberry pi with touching a metal section

            # Predict the stress emotion
            final_result_dict = pr.predict_upload_and_time_specified(signal=signal, gender=gender)
            print("The stress prediction: ", final_result_dict[confv.class_stressed])
            print(final_result_dict)

            None

        else:
            None
            print("Not complied - main")

        # Clean and empty the prerecorded-realtime-specified directory
        dfc.clean_relevant_directory(directory=confv.prerecorded_time_specified_dir)

    else:
        None
        print("Empty - main")


# find_devices()
main()
