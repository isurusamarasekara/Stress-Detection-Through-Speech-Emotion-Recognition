import backbone.support.configurations_variables as confv
import backbone.support.data_loading as dl
import librosa
import backbone.support.plots_and_charts as pc
import matplotlib.pyplot as plt
import numpy as np
import backbone.support.calculations as calc
import backbone.support.configurations_methods as confm
from python_speech_features import logfbank, mfcc


def base_audio_wave_analysis(audio_fname, database, status, gender=confv.gender_nonconformity):
    audio_file_path = dl.get_audio_file_path(audio_fname, database=database, status=status, gender=gender)
    signal, rate = librosa.load(audio_file_path, sr=None)
    print("\tAudio file path: \t", audio_file_path)
    print("\tSample point data of the audio signal: \t", signal)
    print("\tOriginal sampling rate: \t", rate)
    print("\tSignal array dimensions: \t", signal.ndim)
    print("\tShape of the audio signal sample array: \t", signal.shape)
    print("\tNumber of total samples obtained from the audio file: \t", signal.shape[0])
    print("\tLength of the audio file: \t", signal.shape[0] / rate)

    pc.plot_single_audio_wave(signal)
    pc.plot_single_audio_amplitude(signal, rate)
    pc.plot_single_audio_fft(signal, rate)
    pc.plot_single_audio_fft(signal, rate, type=2)
    plt.show()


# In clean samples, the resampling and envelope doesnt matter
# Can remove the dataset condition.
def visual_analysis(df, database, status, gender, envelope=True, resample=True):
    df1 = df.copy()
    classes = list(np.unique(df1.stress_emotion))

    signals = {}
    ffts = {}
    fbanks = {}
    mfccs = {}
    for c in classes:
        if database == confv.database_ravdess:
            aud_fl_pth = dl.get_audio_file_path(audio_fname=df1[df1.stress_emotion == c].iloc[0, 0], database=database, status=status, gender=gender)
            print(aud_fl_pth)
            if resample:
                signal, rate = librosa.load(aud_fl_pth, sr=confv.resample_rate)
            else:
                signal, rate = librosa.load(aud_fl_pth, sr=None)

        elif database == confv.database_emodb:
            aud_fl_pth = dl.get_audio_file_path(audio_fname=df1[df1.stress_emotion == c].iloc[0, 0], database=database, status=status, gender=gender)
            print(aud_fl_pth)
            if resample:
                signal, rate = librosa.load(aud_fl_pth, sr=confv.resample_rate)
            else:
                signal, rate = librosa.load(aud_fl_pth, sr=None)

        elif database == confv.database_cremad:
            aud_fl_pth = dl.get_audio_file_path(audio_fname=df1[df1.stress_emotion == c].iloc[0, 0], database=database, status=status, gender=gender)
            print(aud_fl_pth)
            if resample:
                signal, rate = librosa.load(aud_fl_pth, sr=confv.resample_rate)
            else:
                signal, rate = librosa.load(aud_fl_pth, sr=None)

        elif database == confv.database_shemo:
            aud_fl_pth = dl.get_audio_file_path(audio_fname=df1[df1.stress_emotion == c].iloc[0, 0], database=database, status=status, gender=gender)
            print(aud_fl_pth)
            if resample:
                signal, rate = librosa.load(aud_fl_pth, sr=confv.resample_rate)
            else:
                signal, rate = librosa.load(aud_fl_pth, sr=None)

        if envelope:
            mask = calc.envelope(signal=signal, threshold=confm.get_evelope_threshold(database=database, gender=gender))
            signal = signal[mask]

        signals[c] = signal
        ffts[c] = calc.calc_fft(signal, rate)

        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1200).T
        fbanks[c] = bank

        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1200).T
        mfccs[c] = mel

    title = "Database: " + database + " - " + "Gender: " + gender + " - " + "Envelop: " + str(envelope) + " - " + "Resample: " + str(resample)
    pc.plot_signals(signals, title=title)
    pc.plot_fft(ffts, title=title)
    pc.plot_fbank(fbanks, title=title)
    pc.plot_mfccs(mfccs, title=title)
    plt.show()
