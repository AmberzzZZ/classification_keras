from keras.layers import Input, Conv2D, concatenate, MaxPool2D, Dropout, \
                         GlobalAveragePooling2D, Dense, add, ReLU
from keras.models import Model


def squeezenet(input_shape=(224,224,3), input_tensor=None, n_classes=1000,
               base_filters=128, freq=2, incre=128, sr=0.125, pct=0.5, bypass_mode='simple'):
    if input_tensor is not None:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    # backbone
    x = Conv2D(96, kernel_size=7, strides=2, padding='same', activation='relu')(inpt)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = fire_module(x, 16, 64, 64, bypass_mode)          # fire2
    x = fire_module(x, 16, 64, 64, bypass_mode)          # fire3
    x = fire_module(x, 32, 128, 128, bypass_mode)        # fire4

    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = fire_module(x, 32, 128, 128, bypass_mode)        # fire5
    x = fire_module(x, 48, 192, 192, bypass_mode)        # fire6
    x = fire_module(x, 48, 192, 192, bypass_mode)        # fire7
    x = fire_module(x, 64, 256, 256, bypass_mode)        # fire8

    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = fire_module(x, 64, 256, 256, bypass_mode)        # fire9
    x = Dropout(0,5)(x)
    x = Conv2D(1000, kernel_size=1, strides=1, padding='same', activation='relu')(x)

    # head
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inpt, x)

    return model


def fire_module(x, s_1, e_1, e_3, bypass_mode=None):
    inpt = x
    in_channels = x._keras_shape[-1]
    # squeeze
    x = Conv2D(s_1, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    # expand
    x1 = Conv2D(e_1, kernel_size=1, strides=1, padding='same')(x)
    x2 = Conv2D(e_3, kernel_size=3, strides=1, padding='same')(x)
    # concate
    x = concatenate([x1,x2], axis=-1)
    if bypass_mode:
        if bypass_mode == 'simple' and in_channels==(e_1+e_3):
            x = add([inpt, x])
        if bypass_mode == 'complex':
            if in_channels==(e_1+e_3):
                x = add([inpt, x])
            else:
                inpt = Conv2D((e_1+e_3), kernel_size=1, strides=1, padding='same')(x)
                x = add([inpt, x])
    x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = squeezenet(input_shape=(224,224,3), input_tensor=None, n_classes=1000)
    model.summary()


