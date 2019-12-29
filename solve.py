import os
import variables
import random
from train import train
from predict import predict
from keras.models import load_model
import IPython.display as ipd
import numpy as np

def writeInFile(file, fname, label):
    sep = ','
    
    dataLen = len(fname)

    for i in range(0, dataLen):
        rowText = str(fname[i]) + str(sep) + str(label[i]) + str('\n')
        file.write(rowText)

def solve(x_val, y_val, classes, model):
    file = open('./results.csv',"w")
    firstRow = 'fname,label\n'
    file.write(firstRow)
    fname = []
    label = []
    waves = [f for f in os.listdir(variables.test_path) if f.endswith('.wav')]
    for wav in waves:
        try:
            _, samples = wavfile.read(variables.test_path + wav)
            ipd.Audio(samples, rate=8000)
            fname.append(wav)
            label.append(predict(samples, model, classes))
        except:
            print(variables.test_path + wav + ' cannot be readed.')
    writeInFile(file, fname, label)
    print(fname, label)

if __name__ == "__main__":
    x_val, y_val, classes, model = train()
    solve(x_val, y_val, classes, model)