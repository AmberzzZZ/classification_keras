from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, MaxPool2D, Dropout, ReLU, Softmax


def NiN(input_shape):

    inpt = Input(input_shape)

    x = inpt

    x = mlpConv(x, n_filters=[192,160,96], kernel_size=5, pool_size=3)
    x = mlpConv(x, n_filters=[192,192,192], kernel_size=5, pool_size=3)
    x = mlpConv(x, n_filters=[192,192,10], kernel_size=3, pool_size=3, maxpooling=False)

    x = GlobalAveragePooling2D()(x)
    x = Softmax()(x)

    model = Model(inpt, x)

    return model


def mlpConv(x, n_filters, kernel_size=5, pool_size=3, maxpooling=True):
    # conv
    x = Conv2D(n_filters[0],kernel_size=kernel_size,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # mlp
    for i in range(2):
        x = Conv2D(n_filters[1+i],kernel_size=1,padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    if maxpooling:
        x = MaxPool2D(pool_size=pool_size, strides=2, padding='same')(x)
        x = Dropout(0.5)(x)

    return x


if __name__ == '__main__':

    model = NiN((28,28,1))
    model.summary()



