from keras.layers import Input, Dense, BatchNormalization, Activation, Reshape, UpSampling2D, Conv2D, LeakyReLU, \
                         Dropout, Flatten
from keras.models import Model
import cv2
import numpy as np



# generator model
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


generator.load_weights("generator_40.h5")
noise_tr = np.random.uniform(0,1,size=[1,100])
pred = generator.predict(noise_tr)
print(pred.shape)
img = cv2.resize(pred[0,:,:,0], (100,100))
cv2.imshow("tmp", img)
cv2.waitKey(0)
