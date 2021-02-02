from matplotlib import pyplot as plt


def plot_audio(signal, loudness_threshold=0):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
    plt.plot(signal, color='blue')
    ax.set_xlim((0, len(signal)))

    plt.axhline(y=loudness_threshold, color='r', linestyle='-')

    plt.show()