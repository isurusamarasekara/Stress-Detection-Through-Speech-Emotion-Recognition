from speech_analysis_raspi.support import configurations_variables as confv
import pyaudio
from speech_analysis_raspi.support import recording_configurations as rconf
from speech_analysis_raspi.support import realtime_predict as rpr
import numpy as np
import time
import noisereduce as nr
from speech_analysis_raspi.support import calculations as calc
from datetime import datetime
from speech_analysis_raspi.support import plotting as pl
from speech_analysis_raspi.support import directory_file_checking as dfc


def get_noise_sample_real_time(stream):
    print("Noise data gathering started... For {time} seconds".format(time=rconf.noisy_sample_secs_realtime))

    # Noise Calculation
    raw_noisy_data = stream.read(num_frames=rconf.noisy_sample_num_frames_realtime)     # Would I be open to looping to collect chunksize frames and joining them as in main?
    readable_noisy_sample = np.frombuffer(raw_noisy_data, dtype=rconf.NP_DATATYPE)

    # pl.plot_audio(signal=readable_noisy_sample, loudness_threshold=confv.real_time_noisy_envelope_threshold)
    isComplied, readable_noisy_sample = dfc.check_for_minimum_length_compliance(signal=readable_noisy_sample, threshold=confv.real_time_noisy_envelope_threshold, application_type=confv.app_type_real_time)

    if not isComplied:
        print("Noise data requirement is not fulfilled - considered as silent - assumed no noise is in the background")
        return None, rconf.THRESHOLD_LVL_realtime

    else:
        # pl.plot_audio(signal=readable_noisy_sample, loudness_threshold=confv.real_time_noisy_envelope_threshold)

        # If loudness threshold is determined based on the noisy sample, the final threshold must be returned from here for futher calculations
        # loudness_threshold = np.mean(np.abs(noise_sample)) * rconf.THRESHOLD_LVL_realtime
        loudness_threshold = rconf.THRESHOLD_LVL_realtime
        # pl.plot_audio(signal=readable_noisy_sample, loudness_threshold=loudness_threshold)

        print("Noise data gathering finished - requirement satisfied")

        return readable_noisy_sample, loudness_threshold


def main():
    # User selects the gender
    gender = confv.gender_male

    # Define the databases and their modes to utilize
    database_dict = {
        confv.database_ravdess: confv.ml_mode_convolutional,
        confv.database_shemo: confv.ml_mode_convolutional,
    }
    # print(database_dict)

    # Load model helper onto memory - ModelConfigs, Interpreters and Validation Accuracies
    dataset_helpers_dict = rpr.load_all_helpers_dict(database_dict=database_dict, gender=gender)
    # print(dataset_helpers_dict)
    # for x in dataset_helpers_dict:
    #     print(x)
    #     print(dataset_helpers_dict[x].modelconfig)
    #     print(dataset_helpers_dict[x].modelconfig.database)
    #     print(dataset_helpers_dict[x].modelconfig.gender)
    #     print(dataset_helpers_dict[x].modelconfig.mode)
    #     print(dataset_helpers_dict[x].interpreter)
    #     print(dataset_helpers_dict[x].val_acc)

    p = pyaudio.PyAudio()
    stream = p.open(format=rconf.FORMAT, channels=rconf.CHANNELS, rate=rconf.RATE_realtime, frames_per_buffer=rconf.CHUNKSIZE_realtime, input=True)

    # Get noisy sample - Can even apply a technique to collect and save this noisy file to be referred later universally, like in main_prerecorded_time_specified_noise.
    # Then check for the file existence and if exist, load it rather than collecting noise data.
    # But it will have its own problems.
    noisy_sample, loudness_threshold = get_noise_sample_real_time(stream)
    # print('noisy_sample', noisy_sample)
    # print('noisy_sample shape', noisy_sample.shape)
    # print('loudness_threshold', loudness_threshold)

    time.sleep(rconf.number_of_secs_to_sleep_real_time)
    audio_buffer = []
    minisecond_counter = 0

    file = open(confv.realtime_predictions_file, 'a')

    print("Started realtime recording and prediction...")
    while True:
        minisecond_counter = minisecond_counter + 1

        dateTime = time.asctime(time.localtime(time.time()))

        # Read chunk and load it into numpy array.
        # print("1/10 of second #:", minisecond_counter)

        raw_audio_data_buffer = stream.read(num_frames=rconf.CHUNKSIZE_realtime, exception_on_overflow=False)    # Read per buffer. Buffer is for chunksize.
        timestamp = time.time()
        current_readable_audio_data_window = np.frombuffer(raw_audio_data_buffer, dtype=rconf.NP_DATATYPE)
        # print('current_readable_audio_data_window', current_readable_audio_data_window)
        # print('current_readable_audio_data_window shape', current_readable_audio_data_window.shape)

        '''
        current_reduced_audio_data_window = nr.reduce_noise(audio_clip=current_readable_audio_data_window, noise_clip=noisy_sample, verbose=False)
        if np.mean(np.abs(current_reduced_audio_data_window)) < loudness_threshold:
            print("Silent")
        else:
            print("Active")
        '''

        if len(audio_buffer) == 0:
            audio_buffer = current_readable_audio_data_window
            # print('==audio_buffer', audio_buffer)
            # print('==audio_buffer shape', audio_buffer.shape)
        elif len(audio_buffer) > 0:
            audio_buffer = np.concatenate((audio_buffer, current_readable_audio_data_window), axis=None)
            # print('>audio_buffer', audio_buffer)
            # print('>audio_buffer shape', audio_buffer.shape)

        if minisecond_counter == rconf.counter_threshold_realtime:
            # print('minisecond_counter', minisecond_counter)
            # print('audio_buffer', audio_buffer)
            # print('audio_buffer shape', audio_buffer.shape)
            # pl.plot_audio(signal=audio_buffer, loudness_threshold=loudness_threshold)

            if noisy_sample is not None:
                noise_reduced = nr.reduce_noise(audio_clip=audio_buffer, noise_clip=noisy_sample, verbose=False)
                # print('noise_reduced_final', noise_reduced)
                # print('noise_reduced_final shape', noise_reduced.shape)
                # pl.plot_audio(signal=noise_reduced, loudness_threshold=loudness_threshold)

                audio_buffer = noise_reduced

            # Minimum signal length check
            # For below to be the condition, step_size should be same as in the build features in model training
            # step_size is bound to build features in model training.
            isComplied, final_signal = dfc.check_for_minimum_length_compliance(signal=audio_buffer, threshold=loudness_threshold, application_type=confv.app_type_real_time)

            # print('final_signal', final_signal)
            # print('final_signal shape', final_signal.shape)

            if isComplied:
                # If not complied, having this plot outside of the conditional statement will have a blank plot
                # pl.plot_audio(signal=final_signal, loudness_threshold=loudness_threshold)

                # Predict the stress emotion
                final_result = rpr.predict_real_time(dataset_helpers_dict=dataset_helpers_dict, signal=final_signal)

                print('Datetime: ', datetime.fromtimestamp(timestamp), ' - Final Predictions:', final_result)

                if final_result[confv.class_stressed] > 0.5:
                    file.write(dateTime + ' ' + 'stressed' + ' ' + str(final_result[confv.class_stressed]) + '\n')
                    file.flush()
                else:
                    file.write(dateTime + ' ' + 'not-stressed' + ' ' + str(final_result[confv.class_stressed]) + '\n')
                    file.flush()

            else:
                print("Considering the whole audio_buffer as silent")
                file.write(dateTime + ' ' + 'silent' + '\n')
                file.flush()
                # Record in text file as silent

            audio_buffer = []
            minisecond_counter = 0


main()
