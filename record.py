import IPython.display as ipd
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import random
import os
from train import train
from predict import predict
from keras.models import load_model

def record(x_val, y_val, classes):
    samplerate = 16000  
    duration = 1
    filename = './voice_commands/myvoice.wav'
    print("Press enter when you are ready to speak")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
    sd.wait()
    sf.write(filename, mydata, samplerate)

    model = load_model('best_model.hdf5')
    samples, sample_rate = librosa.load(filename, sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ipd.Audio(samples, rate=8000)
    predict(samples, model, classes)

if __name__ == "__main__":
    x_val, y_val, classes = train()
    record(x_val, y_val, classes)