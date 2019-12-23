import variables
import os
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np

def resample_data():
    labels = [d for d in os.listdir(variables.train_path) if os.path.isdir(variables.train_path + d)]
    all_waves = []
    all_labels = []
    for label in labels:
        print("Processing ", label)
        waves = [f for f in os.listdir(variables.train_path + label) if f.endswith('.wav')]
        for wav in waves:
            samples, sample_rate = librosa.load(variables.train_path + label + '/' + wav, sr=16000)
            samples = librosa.resample(samples, sample_rate, 8000)
            if (len(samples) == 8000) : 
                all_waves.append(samples)
                all_labels.append(label)
    print("All done!")
    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    classes = list(le.classes_)
    y = np_utils.to_categorical(y, num_classes=len(labels))
    all_waves = np.array(all_waves).reshape(-1, 8000, 1)
    return all_waves, y, labels, classes
