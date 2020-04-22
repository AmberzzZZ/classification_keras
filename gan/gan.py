#!/usr/bin/env python
# Keras GAN Implementation
import os
import glob
import random
import numpy as np
import cv2
from keras.optimizers import Adam
from keras.layers import Input, Dense, BatchNormalization, Activation, Reshape, UpSampling2D, Conv2D, LeakyReLU, \
                         Dropout, Flatten
from keras.models import Model



from keras.utils import np_utils
from keras.layers import Input,merge
from keras.layers import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
import matplotlib.pyplot as plt
from keras.utils import np_utils
from tqdm import tqdm


######### data ###########
img_rows, img_cols = 28, 28
train_path = "../train"
X = []
Y = []
for cls_folder in os.listdir(train_path):
    if cls_folder not in ['d0', 'd1']:
        continue
    cls = int(cls_folder[-1])
    for file in [i for i in glob.glob(os.path.join(train_path, cls_folder) + "/*")][:120]:
        image = cv2.imread(file, 0)
        X.append(image)
        Y.append(cls)
idx = random.shuffle([i for i in range(100)])
X_train = np.array(X[:100])[idx].reshape((100, img_rows, img_cols, 1)) / 255.
Y_train = np.array(Y[:100])[idx]
idx = random.shuffle([i for i in range(20)])
X_test = np.array(X[100:120])[idx].reshape((20, img_rows, img_cols, 1)) / 255.
Y_test = np.array(Y[100:120])[idx]

print(np.min(X_train), np.max(X_train))
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


########## model ##########
shp = X_train.shape[1:]
dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

# Build Generative model ...
nch = 200
g_input = Input(shape=[100])
H = Dense(nch*14*14, init='glorot_normal')(g_input)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Reshape([14, 14, nch])(H)
H = UpSampling2D(size=(2, 2))(H)
H = Conv2D(nch//2, 3, padding='same', init='glorot_uniform')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Conv2D(nch//4, 3, padding='same', init='glorot_uniform')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Conv2D(1, 1, padding='same', init='glorot_uniform')(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
# generator.summary()

# Build Discriminative model ...
d_input = Input(shape=shp)
H = Conv2D(256, 5, strides=2, padding='same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Conv2D(512, 5, strides=2, padding='same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2,activation='softmax')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
# discriminator.summary()

# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
make_trainable(discriminator, False)

# Build stacked GAN model
gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
# GAN.summary()


######## train ########
# Pre-train the discriminator network ...
noise_gen = np.random.uniform(0,1,size=[X_train.shape[0],100])
generated_images = generator.predict(noise_gen)
print("shape: ", generated_images.shape , X_train.shape)
X = np.concatenate((X_train, generated_images))
n = X_train.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1

make_trainable(discriminator,True)
discriminator.fit(X,y, nb_epoch=1, batch_size=128)
y_hat = discriminator.predict(X)


# Set up our main training loop
def train_for_n(nb_epoch=5000, BATCH_SIZE=32):

    for e in tqdm(range(nb_epoch)):

        # Make generative images
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_images = generator.predict(noise_gen)

        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1

        make_trainable(discriminator,True)
        d_loss = discriminator.train_on_batch(X,y)

        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1

        make_trainable(discriminator,False)
        g_loss = GAN.train_on_batch(noise_tr, y2)

        if e and e%20==0:
            generator.save_weights("generator_%d.h5" % e)

    return d_loss, g_loss


# Train for 6000 epochs at original learning rates
train_for_n(nb_epoch=200, BATCH_SIZE=32)

# Train for 2000 epochs at reduced learning rates ...




