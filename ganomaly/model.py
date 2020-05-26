from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Conv2DTranspose, ReLU, Activation, Lambda
from keras.models import Model
from keras.optimizers import adam
import keras.backend as K


def l1_loss(inputs):
    y_true, y_pred = inputs
    return K.mean(K.abs(y_true - y_pred))


def l2_loss(inputs):
    y_true, y_pred = inputs
    return K.mean(K.pow(y_true - y_pred, 2))


def encoder(input_shape=(128,128,1)):
    inpt = Input(input_shape)

    x = Conv2D(64, 4, strides=2, padding='same')(inpt)
    x = LeakyReLU(0.2)(x)

    while x._keras_shape[1] > 4:
        x = Conv2D(x._keras_shape[-1]*2, 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    x = Conv2D(100, 4, strides=1, padding='valid')(x)

    model = Model(inpt, x)

    return model


def decoder(input_shape=(1,1,100)):
    inpt = Input(input_shape)

    x = Conv2DTranspose(1024, 4, strides=1, padding='valid')(inpt)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    factor = 2
    while factor <= 16:
        x = Conv2DTranspose(1024//factor, 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        factor *= 2

    x = Conv2DTranspose(1, 4, strides=2, padding='same')(x)
    x = Activation('tanh')(x)

    model = Model(inpt, x)

    return model


def NetD(input_shape=(128,128,1), lr=3e-4, decay=5e-6):
    en = encoder(input_shape=input_shape)
    features, vector = en.get_layer(index=-2).output, en.get_layer(index=-1).output
    x = Activation('sigmoid')(vector)
    dis_model = Model(en.input, [features, x])

    x_real = Input(input_shape)
    f_real, vec_real = dis_model(x_real)
    x_fake = Input(input_shape)
    f_fake, vec_fake = dis_model(x_fake)

    l2_adv = Lambda(l2_loss)([f_real,f_fake])

    model = Model([x_real, x_fake], l2_adv)
    model.compile(adam(lr,decay),
                  loss=lambda y_true, y_pred:y_pred,
                  metrics=None)

    return model


def NetG(input_shape=(128,128,1), w_con=50, w_enc=1, lr=3e-4, decay=5e-6):
    en1 = encoder(input_shape=input_shape)
    de = decoder(input_shape=en1.output._keras_shape[1:])
    en2 = encoder(input_shape=input_shape)

    x = Input(input_shape)
    x1 = en1(x)
    x2 = de(x1)
    x3 = en2(x2)

    l1_con = Lambda(l1_loss)([x, x2])
    l2_enc = Lambda(l2_loss)([x1,x3])

    model = Model(x, [l1_con, l2_enc])

    model.compile(adam(lr,decay),
                  loss=lambda y_true, y_pred:y_pred, loss_weights=[w_con, w_enc],
                  metrics=None)

    return model


if __name__ == '__main__':

    # en = encoder(input_shape=(128,128,1))
    # en.summary()

    # de = decoder(input_shape=(1,1,100))
    # de.summary()

    # netD = NetD(input_shape=(128,128,1))
    # netD.summary()

    netG = NetG(input_shape=(128,128,1))
    netG.summary()







