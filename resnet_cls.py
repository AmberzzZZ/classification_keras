import os
from keras.layers import Input, BatchNormalization, Conv2D, GlobalAveragePooling2D,  \
                         Dense, add, Lambda
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.backend import tf as K
import numpy as np
import random

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

data_pt = "./"
weight_pt = "./weights/"


class PowerTransferMode:

    def DataGen(self, data_pt, img_size, batch_size, is_train):
        if is_train:
            datagen = ImageDataGenerator(rescale=None)
        else:
            datagen = ImageDataGenerator(rescale=None)

        generator = datagen.flow_from_directory(
            data_pt, target_size=(img_size, img_size),
            color_mode='grayscale',
            batch_size=batch_size)

        return generator

    def Datagen_mixup(self, data_path, img_size, batch_size, is_train=True, mix_prop=0.8, alpha=1.0):
        if is_train:
            datagen = ImageDataGenerator()
        else:
            datagen = ImageDataGenerator()

        generator = datagen.flow_from_directory(
            data_path, target_size=(img_size, img_size),
            batch_size=batch_size,
            color_mode="grayscale",
            shuffle=True)

        for x,y in generator:     # a batch of <img, label>
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1
            idx = [i for i in range(x.shape[0])]
            random.shuffle(idx)
            mixed_x = lam*x + (1-lam)*x[idx]
            mixed_y = lam*y + (1-lam)*y[idx]

            n_origin = int(batch_size * mix_prop)
            gen_x = np.vstack((x[:n_origin], mixed_x[:(batch_size-n_origin)]))
            gen_y = np.vstack((y[:n_origin], mixed_y[:(batch_size-n_origin)]))

            yield gen_x, gen_y

    def Conv2d_BN(self, x, nb_filter, kernel_size, strides=(1,1), padding='same', name=None):
        bn_name = name + '_bn' if name else None
        conv_name = name + '_conv' if name else None

        x = Conv2D(filters=nb_filter, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(self, inpt, nb_filter, kernel_size, strides=(1,1), with_conv_shortcut=False):
        x = self.Conv2d_BN(inpt, nb_filter, kernel_size, strides, padding='same')
        x = self.Conv2d_BN(x, nb_filter, kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = self.Conv2d_BN(inpt, nb_filter, kernel_size, strides, padding='same')
            x = add([x, shortcut])
        else:
            x = add([x, inpt])
        return x

    # resnet18
    def resnet18_model(self, lr=3e-4, decay=1e-6, nb_classes=2, img_size=512, target_size=64, RGB=True):
        color = 3 if RGB else 1
        print('color channel: ', color)

        inpt = Input(shape=(img_size, img_size, color))
        x = Lambda(lambda image: K.image.resize_images(image, (target_size, target_size)))(inpt)

        x = self.Conv2d_BN(x, nb_filter=64, kernel_size=(3,3), strides=(1,1))

        # 64,64
        x = self.Conv_Block(x, nb_filter=64, kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
        x = self.Conv_Block(x, nb_filter=64, kernel_size=(3,3))

        # 32,32
        x = self.Conv_Block(x, nb_filter=64, kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
        x = self.Conv_Block(x, nb_filter=64, kernel_size=(3,3))

        # 16,16
        x = self.Conv_Block(x, nb_filter=64, kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
        x = self.Conv_Block(x, nb_filter=64, kernel_size=(3,3))

        # 8,8
        x = self.Conv_Block(x, nb_filter=64, kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
        x = self.Conv_Block(x, nb_filter=64, kernel_size=(3,3))

        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)

        sgd = SGD(lr=lr, decay=decay)
        model.compile(optimizer=sgd,
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        return model

    def resnet50_model(self, lr=3e-4, decay=1e-6, nb_classes=2, img_size=512, target_size=224, RGB=True,
                       weights=None, include_top=False):
        color = 3 if RGB else 1
        print('color channel: ', color)

        inpt = Input(shape=(img_size, img_size, color))
        x = Lambda(lambda image: K.image.resize_images(image, (target_size, target_size)))(inpt)

        base_model = ResNet50(include_top=include_top, weights=weights, input_tensor=x)
        x = base_model.outputs[0]

        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        model.summary()

        return model

    def train_model(self, model, epochs,
                    train_generator, steps_per_epoch,
                    validation_generator, validation_steps,
                    model_url, is_load_model=True):
        if is_load_model and os.path.exists(model_url):
            print("load model: ", model_url)
            model.load_weights(model_url, by_name=True)

        filepath = weight_pt + "/r18_weights_{epoch:02d}_val_acc_{val_acc:.3f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        # class_weights = {0:1, 1:1, 2:1.5}
        # print("class_weights: ", class_weights)
        history_ft = model.fit_generator(generator=train_generator,
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=epochs,
                                         validation_data=validation_generator,
                                         validation_steps=validation_steps,
                                         callbacks=callbacks_list,
                                         # class_weights=class_weights,
                                         verbose=1)
        return history_ft


if __name__ == "__main__":
    image_size = 512
    target_size = 64
    batch_size = 32
    n_classes = 2
    n_epoch = 5

    transfer = PowerTransferMode()

    # data
    train_generator = transfer.Datagen_mixup(data_pt + "/train", image_size, batch_size)
    val_generator = transfer.Datagen_mixup(data_pt + "/train", image_size, batch_size, is_train=False)

    # model
    model = transfer.resnet18_model(lr=3e-4, decay=1e-6, nb_classes=2,
                                    img_size=image_size, target_size=target_size, RGB=False)

    # model = transfer.resnet50_model(lr=3e-4, decay=1e-6, nb_classes=2,
    #                                 img_size=image_size, RGB=False)

    # train
    history_ft = transfer.train_model(model, n_epoch, train_generator, 10, val_generator, 2,
                                      'r18_weights')




















