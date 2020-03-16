from keras.models import Model
from keras.layers import Input, Conv2D, Add, UpSampling2D, GlobalAveragePooling2D, Dense
from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model
import keras.backend as K


def backbone(input_tensor=None):
    if input_tensor is None:
        inpt = Input((None,None,3))
    else:
        inpt = input_tensor
    # bottom-up pathway
    model = ResNet50(include_top=False, weights=None, pooling=None, input_tensor=input_tensor)
    features = ['activation_1', 'activation_10', 'activation_22', 'activation_40', 'activation_49']
    outputs = Model(model.input, [model.get_layer(features[i]).output for i in range(5)])(inpt)
    return outputs


def fpn(input_tensor=None, num_classes=10):
    if input_tensor is None:
        inpt = Input((None,None,3))
    else:
        inpt = input_tensor
    convs = backbone(inpt)
    _, C2, C3, C4, C5 = convs
    # top-down pathway & lateral connection
    P5 = Conv2D(256, kernel_size=(1,1), name="fpn_c5p5")(C5)
    P4 = Add(name="fpn_p4add")([UpSampling2D(name="fpn_p5Up")(P5), Conv2D(256, kernel_size=(1,1), name="fpn_c4p4")(C4)])
    P3 = Add(name="fpn_p3add")([UpSampling2D(name="fpn_p4Up")(P4), Conv2D(256, kernel_size=(1,1), name="fpn_c3p3")(C3)])
    P2 = Add(name="fpn_p2add")([UpSampling2D(name="fpn_p3Up")(P3), Conv2D(256, kernel_size=(1,1), name="fpn_c2p2")(C2)])
    # head
    P5 = Conv2D(256, kernel_size=(3,3), padding='same', name="fpn_p5")(P5)
    P4 = Conv2D(256, kernel_size=(3,3), padding='same', name="fpn_p4")(P4)
    P3 = Conv2D(256, kernel_size=(3,3), padding='same', name="fpn_p3")(P3)
    P2 = Conv2D(256, kernel_size=(3,3), padding='same', name="fpn_p2")(P2)
    # shared classifier for example
    shared = Dense(num_classes, activation='softmax')
    model = Model(inpt, [shared(i) for i in [P5, P4, P3, P2]])

    return model


if __name__ == '__main__':

    input_tensor = Input((256,256,3))
    model = fpn(input_tensor)
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')
    model.summary()





