# coding: utf-8

import matplotlib.pyplot as plt
import os
from keras.utils import plot_model
from keras.layers import Input, Conv2D, BatchNormalization, add, Dense, AveragePooling2D, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import glob
import numpy as np
import cv2


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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


def vgg19_model(lr=3e-4, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=224, img_cols=224, RGB=True, is_plot_model=False):
    color = 3 if RGB else 1
    print("number of channels: ", color)

    base_model = VGG19(weights=None, include_top=False, pooling=None, input_shape=(img_rows, img_cols, color), classes=nb_classes)

    x = base_model.output
    x = AveragePooling2D(pool_size=(7,7))(x)
    x = Flatten()(x)
    fv = x
    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(input=base_model.input, output=[x, fv])

    for layer in base_model.layers[:]:
        layer.trainable = False

    sgd = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy', recall, precision])

    if is_plot_model:
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

    # model.save('vgg19model.h5')
    return model


def train_model(model, epochs, train_generator, steps_per_epoch, validation_generator, validation_steps,model_url, is_load_model):
    if is_load_model and os.path.exists(model_url):
        print(model_url)
        model.load_weights(model_url)

    filepath = "v19_weights-improvement-{epoch:02d}-{val_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,mode='auto')
    callbacks_list = [checkpoint]
    class_weight = {0:3, 1:1, 2:1.5, 3:1}
    print("class weight: ", class_weight)
    history_ft = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        class_weight=class_weight,
        verbose=1)

    return history_ft


def plot_training(history, n_epoch):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    # acc
    plt.plot(epochs, acc, 'b-', label='acc')
    plt.plot(epochs, val_acc, 'r-', label='val_acc')
    plt.title('Training and validation accracy')
    plt.savefig(str(n_epoch) + 'acc')
    # loss
    plt.plot(epochs, loss, 'b-', label='loss')
    plt.plot(epochs, val_loss, 'r-', label='val_loss')
    plt.title('Training and validation loss')
    plt.savefig(str(n_epoch) + 'loss')


def load_train(img_lst, nb_classes, width, height, batch_size, rgb):
    while True:
        images = []
        labels = []
        number = np.random.random_integers(0, len(img_lst)-1, batch_size)
        for i in number:
            path = img_lst[i][0]
            label = img_lst[i][1]
            image_data = cv2.imread(path, cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)
            image_data = cv2.resize(image_data,(width,height),0,0,cv2.INTER_LINEAR)
            image_data = image_data.astype(np.float32)
            if not rgb:
                image_data = np.reshape(image_data, (width, height, -1))
            image_data = np.multiply(image_data, 1.0 / 255.0)
            images.append(image_data)
            tmp = np.zeros(nb_classes)
            tmp[label] = 1
            labels.append(tmp)
        images = np.array(images)
        labels = np.array(labels)
        yield images, labels


if __name__ == '__main__':
    image_size = 224
    batch_size = 20
    nb_classes = 2
    rgb = True
    n_epoch = 3

    train_path = "./train"
    cate = [i for i in glob.glob(train_path+"/*") if os.path.isdir(i)]
    img_lst = []
    for idx, each_cls in enumerate(cate):
        imgs = glob.glob(each_cls+"/*")
        lbs = [idx] * len(imgs)
        img_lst += zip(imgs, lbs)
    print("detecting %d images belonging to %d classes" % (len(imgs), len(cate)))

    val_path = "./val"

    # generator
    train_generator = load_train(img_lst, nb_classes, image_size, image_size, batch_size, rgb)
    validation_generator = load_train(img_lst, nb_classes, image_size, image_size, batch_size, rgb)

    # model
    model = vgg19_model(nb_classes=nb_classes, img_rows=image_size, img_cols=image_size, RGB=rgb, is_plot_model=True)

    # fit generator
    history_ft = train_model(model, n_epoch, train_generator, 1, validation_generator, 1, 'v19', is_load_model=False)

    # plot_training(history_ft, n_epoch)



