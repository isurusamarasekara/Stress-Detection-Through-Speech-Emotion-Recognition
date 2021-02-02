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
import noisereduce as nr
from speech_analysis_raspi.support import calculations as calc


def record(time_to_record, record_type, audio_save_path=""):
    if record_type == confv.prerecorded_time_specified_noise_record_type_noisy and not audio_save_path:
        sys.exit()

    print("Started recording {record_type} clip... For {time} seconds".format(record_type=record_type, time=time_to_record))
    audio_signal = sd.rec(frames=int(time_to_record * rconf.RATE_prerecorded_time_specified_noise), samplerate=rconf.RATE_prerecorded_time_specified_noise, channels=rconf.CHANNELS, dtype='float32')
    sd.wait()
    audio_signal = audio_signal.flatten()
    print(audio_signal.shape)
    print("Finished recording {record_type} clip...".format(record_type=record_type))
    if record_type == confv.prerecorded_time_specified_noise_record_type_noisy:

        # Before envelope plotting has the red line threshold of noise threshold that will be used to remove the silent sections of the noise clip only
        # pl.plot_audio(signal=audio_signal, loudness_threshold=confv.prerecorded_time_specified_noise_noisy_envelope_threshold)
        isComplied, audio_signal = dfc.check_for_minimum_length_compliance(signal=audio_signal, threshold=confv.prerecorded_time_specified_noise_noisy_envelope_threshold, application_type=confv.app_type_prerecorded_time_specified_noise)
        if not isComplied:
            print("Noisy sample length not complied - considered as silent - assumed no noise is in the background")
            return None
        # After envelope plotting has the red line threshold of that will be used for the required voice clip
        # pl.plot_audio(signal=audio_signal, loudness_threshold=confv.prerecorded_time_specified_noise_envelope_threshold)

        print('Saving noisy audio clip to', audio_save_path)
        sf.write(file=audio_save_path, data=audio_signal, samplerate=rconf.RATE_prerecorded_time_specified_noise)

    elif record_type == confv.prerecorded_time_specified_noise_record_type_required:
        print("playing required clip")
        play_signal(audio_signal)

        pass

    return audio_signal


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


def play_signal(audio_signal):
    wave_obj = sa.WaveObject(audio_signal, 1, 4, 16000)  # 4 bytes per one float32 sample
    play_obj = wave_obj.play()
    play_obj.wait_done()

    return


def main():
    print("============================Started loading, if not Creating, the noisy clip============================")
    # Create the noisy_clips directory if not already created
    dfc.check_dir(directory=confv.prerecorded_time_specified_noisy_clips_dir)

    # Check if the noisy_clip directory is empty
    isEmpty = dfc.is_directory_empty(directory=confv.prerecorded_time_specified_noisy_clips_dir)
    if isEmpty:
        # record and store noisy clip
        noisy_clip_secs = confv.prerecorded_time_specified_noise_seconds
        noisy_clip_aud_fl_pth = confv.prerecorded_time_specified_noisy_clip_file_aud_fl_pth
        raspi_noise_signal = record(time_to_record=noisy_clip_secs, record_type=confv.prerecorded_time_specified_noise_record_type_noisy, audio_save_path=noisy_clip_aud_fl_pth)
        if raspi_noise_signal is not None:
            print("playing noisy clip")
            play(noisy_clip_aud_fl_pth)

            print('raspi_noise_signal', raspi_noise_signal)
            print('raspi_noise_signal shape', raspi_noise_signal.shape)
            # Here the noisy clip is scaled against the threshold level set for the required audio
            # pl.plot_audio(signal=raspi_noise_signal, loudness_threshold=confv.prerecorded_time_specified_noise_envelope_threshold)  # No need for threshold map. It is just for scale understanding
            print("============================Finished Creating, the noisy clip============================")

        else:
            print("============================Noise clip not created============================")

    else:
        # Below loading is done
        raspi_noise_signal, rate = sf.read(file=confv.prerecorded_time_specified_noisy_clip_file_aud_fl_pth)    # Can use librosa, since nor resampling required

        print('raspi_noise_signal', raspi_noise_signal)
        print('raspi_noise_signal shape', raspi_noise_signal.shape)
        # Here the noisy clip is scaled against the threshold level set for the required audio
        # pl.plot_audio(signal=raspi_noise_signal, loudness_threshold=confv.prerecorded_time_specified_noise_envelope_threshold)  # No need for threshold map. It is just for scale understanding
        print("============================Finished loading the noisy clip============================")

    # User selects the gender
    gender = confv.gender_male

    # User input number of seconds to record
    prerecorded_time_specified_noise_required_seconds = 10

    # Recording the required voice clip with the specified time
    recorded_signal = record(time_to_record=prerecorded_time_specified_noise_required_seconds, record_type=confv.prerecorded_time_specified_noise_record_type_required)
    print('recorded_signal', recorded_signal)
    print('recorded_signal shape', recorded_signal.shape)
    # pl.plot_audio(signal=recorded_signal, loudness_threshold=confv.prerecorded_time_specified_noise_envelope_threshold)

    if raspi_noise_signal is not None:
        noise_reduced_final = nr.reduce_noise(audio_clip=recorded_signal, noise_clip=raspi_noise_signal, verbose=False)
        print('noise_reduced_final', noise_reduced_final)
        print('noise_reduced_final shape', noise_reduced_final.shape)
        # pl.plot_audio(signal=noise_reduced_final, loudness_threshold=confv.prerecorded_time_specified_noise_envelope_threshold)
        # play_signal(audio_signal=noise_reduced_final) # Issue with raspi - listen to this. In other platforms, works fine

        recorded_signal = noise_reduced_final

    # Minimum signal length check - envelope
    # For below to be the condition, step_size should be same as in the build features in model training
    # step_size is bound to build features in model training.
    isComplied, signal = dfc.check_for_minimum_length_compliance(signal=recorded_signal, threshold=confv.prerecorded_time_specified_noise_envelope_threshold, application_type=confv.app_type_prerecorded_time_specified_noise)
    print('final signal', signal)
    print('final signal shape', signal.shape)
    # play_signal(signal)
    if isComplied:
        # If not complied, having this plot outside of the conditional statement will have a blank plot
        # pl.plot_audio(signal=signal, loudness_threshold=confv.prerecorded_envelope_threshold)

        # Predict the stress emotion
        final_result_dict = pr.predict_upload_and_time_specified(signal, gender)
        print("The stress prediction: ", final_result_dict[confv.class_stressed])
        print(final_result_dict)

    else:
        print("Not complied - main")


# find_devices()
main()
