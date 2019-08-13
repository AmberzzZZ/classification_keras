from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import load_model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import glob
import random
import cv2
import keras.backend as K


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def lstm_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(28,28)))    # hidden units is a key parameter to tune with
    # model.add(LSTM(50, return_sequences=True))     # multiple layers
    # model.add(LSTM(50))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    rmsprop = RMSprop(lr=0.001, decay=1e-2)     # decay is a key param to tune with
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy', recall, precision]) 
    return model


def cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=input_shape, strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.summary()
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy', recall, precision]) 
    return model


if __name__ == "__main__":
    data_dir = "data"
    cate = [i for i in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, i))]
    print("cate: ", cate)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for idx, each_cls in enumerate(cate):
        target = [0] * len(cate)
        target[idx] = 1
        for img in glob.glob(os.path.join(data_dir, each_cls)+"/*png"):
            cv_img = cv2.imread(img, 0)
            tmp = cv_img / 255.
            x_train.append(tmp)
            y_train.append(target)

    mean = np.mean(x_train, axis=0)
    x_train -= mean
    idx = [i for i in range(len(x_train))]
    random.shuffle(idx)
    x_train = np.array(x_train)[idx]
    y_train = np.array(y_train)[idx]

    # lstm model
    model =lstm_model()

    # cnn model
    # x_train = x_train.reshape((-1,28,28,1))
    # x_test = x_test.reshape((-1,28,28,1))
    # model = cnn_model((28,28,1))

    # train
    filepath = "lstm_weights_{epoch:02d}_acc_{acc:.3f}_val_acc{val_acc.3f}.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_lst = [checkpoint1, checkpoint2]
    model.fit(x=x_train, y=y_train, batch_size=32, epochs=5, 
              verbose=1, callbacks=None, 
              validation_split=0.2, shuffle=True,
              callbacks=callbacks_lst)




