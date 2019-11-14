from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
import random
import itertools
import sys
sys.path.append("../centerloss/")
from dataLoader import load_mnist


def generate_triplet(x,y,ap_pairs=10,an_pairs=10):
    triplet_train_pairs = []
    triplet_p_n_labels = []
    data_xy = tuple([x,y])
    for data_class in np.unique(y):
        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        A_P_pairs = random.sample(list(itertools.permutations(same_class_idx,2)),k=ap_pairs)
        Neg_idx = random.sample(list(diff_class_idx),k=an_pairs)

        for ap in A_P_pairs:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_train_pairs.append([Anchor,Positive,Negative])   # O(N*N*N)
                triplet_p_n_labels.append([data_class, data_xy[1][n]])

    return np.array(triplet_train_pairs), np.array(triplet_p_n_labels)


def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    y_true -- don't actually needed in this function.
    y_pred -- encoding vectors of anchor, positive and negative
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ',y_pred)          # [Batch, vec_dims*3]

    total_lenght = y_pred.shape.as_list()[-1]

    anchor = y_pred[:,0:int(total_lenght*1/3)]
    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)

    return loss


def base_model(input_shape=(28,28,1)):
    input = Input(shape=input_shape)
    x = Conv2D(128,(7,7),padding='same',input_shape=(input_shape[0],input_shape[1],input_shape[2],),activation='relu',name='conv1')(input)
    x = MaxPooling2D((2,2),(2,2),padding='same',name='pool1')(x)
    x = Conv2D(256,(5,5),padding='same',activation='relu',name='conv2')(x)
    x = MaxPooling2D((2,2),(2,2),padding='same',name='pool2')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(4,name='embeddings')(x)
    model = Model(input=input, output=x)
    # plot_model(model, to_file='images/base_model.png', show_shapes=True, show_layer_names=True)

    return model


def triple_model(input_shape=(28,28,1)):
    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    negative_input = Input(shape=input_shape, name='negative_input')

    sharedDNN = base_model(input_shape=(28,28,1))
    encoded_anchor = sharedDNN(anchor_input)
    encoded_positive = sharedDNN(positive_input)
    encoded_negative = sharedDNN(negative_input)
    merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
    # plot_model(model, to_file='images/triple_model.png', show_shapes=True, show_layer_names=True)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    sgd = SGD(lr=0.001, momentum=0.9, decay=0.1, nesterov=True)
    model.compile(loss=triplet_loss, optimizer=adam)

    return model


if __name__ == '__main__':

    x_train, y_train = load_mnist()
    x_train_pairs, _ = generate_triplet(x_train,y_train, ap_pairs=150, an_pairs=150)
    Anchor = x_train_pairs[:,0,:].reshape(-1,28,28,1)
    Positive = x_train_pairs[:,1,:].reshape(-1,28,28,1)
    Negative = x_train_pairs[:,2,:].reshape(-1,28,28,1)
    y_dummy = np.zeros((Anchor.shape[0],1))

    filepath = "./tripleNet_{epoch:02d}_val_loss_{val_loss:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, monitor="val_loss", mode='min', save_best_only=True)

    model = triple_model(input_shape=(28,28,1))
    model.fit(x=[Anchor, Positive, Negative], y=y_dummy, 
              batch_size=8, epochs=100, 
              validation_split=0.2, 
              verbose=1, callbacks=[checkpoint])


