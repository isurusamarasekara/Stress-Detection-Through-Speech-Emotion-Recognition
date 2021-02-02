import pandas as pd
import numpy as np
import backbone.support.configurations_variables as confv


def envelope(signal, threshold):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    y_mean = y.rolling(window=confv.step_size, min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask


def calc_fft(signal, rate):
    n = len(signal)
    # print(rate)
    freq = np.fft.rfftfreq(n, d=1 / rate) # Issue here if envelope threshold is too high in emodb female - 05 fails
    # print(freq)
    Y = abs(np.fft.rfft(signal) / n)
    # print("======================================")
    # print(Y)
    # print("======================================")
    # print(freq)
    return Y, freq
