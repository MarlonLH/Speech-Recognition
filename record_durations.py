import os
import variables
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

def record_durations():
    labels = [d for d in os.listdir(variables.train_path) if os.path.isdir(variables.train_path + d)]
    durations = []
    for label in labels:
        waves = [f for f in os.listdir(variables.train_path + label) if f.endswith('.wav')]
        for wav in waves:
            try:
                sample_rate, samples = wavfile.read(variables.train_path + label + '/' + wav)
                durations.append(float(len(samples) / sample_rate))
            except:
                print(variables.train_path + label + '/' + wav + ' cannot be readed.')
    plt.hist(np.array(durations))
    plt.show()

if __name__ == "__main__":
    record_durations()