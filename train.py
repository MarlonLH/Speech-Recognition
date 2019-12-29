import numpy as np
from display_results import display_results
from sklearn.model_selection import train_test_split
from resample_data import resample_data
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import load_model

def train():
    all_waves, y, labels, classes = resample_data()
    x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_waves),
                                                np.array(y),
                                                stratify=y,
                                                test_size=0.2,
                                                random_state=777,
                                                shuffle=True)
    K.clear_session()
    inputs = Input(shape=(8000,1))
    conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    conv = Flatten()(conv)
    conv = Dense(256, activation='relu')(conv)
    conv = Dropout(0.3)(conv)
    conv = Dense(128, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    outputs = Dense(len(labels), activation='softmax')(conv)

    model = Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
    mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = model.fit(x_tr, y_tr ,epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))
    display_results(history)
    modelCpy = model
    try:
        model = load_model('best_model.hdf5')
    except:
        model = modelCpy
    return x_val, y_val, classes, model

if __name__ == "__main__":
    train()