
import librosa
import matplotlib.pyplot as plt
import numpy as np
import variables

def display_raw_wave(path, rate):
    samples, sr = librosa.load(variables.train_path + path)
    print(sr)
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(rate)
    ax1.set_title('Raw wave')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, sr / len(samples), sr), samples)
    plt.show()

if __name__ == "__main__":
    display_raw_wave('bed/0a7c2a8d_nohash_0.wav', 211)