import random
from train import train
from predict import predict
from keras.models import load_model
import IPython.display as ipd
import numpy as np

def solve(x_val, y_val, classes, model):
    model = load_model('best_model.hdf5')
    index = random.randint(0, len(x_val) - 1)
    samples = x_val[index].ravel()
    print("Audio:", classes[np.argmax(y_val[index])])
    ipd.Audio(samples, rate=8000)
    print("Prediction:", predict(samples, model, classes))

if __name__ == "__main__":
    x_val, y_val, classes, model = train()
    solve(x_val, y_val, classes, model)