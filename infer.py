from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import load_model
import cv2
import glob
import numpy as np


def lstm_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(28,28)))    # hidden units is a key parameter to tune with
    model.add(Dense(2, activation='softmax'))

    return model


if __name__ == "__main__":
    model = lstm_model()
    model.load_weights("filepath")

    datadir = "testdata/"

    for img in glob.glob(datadir+"/*png"):
        cv_img = cv2.imread(img, 0)
        tmp = np.array(cv_img, dtype=np.float32)
        tmp /= 255
        pred = model.predict(tmp)
        cls = np.argmax(pred)
