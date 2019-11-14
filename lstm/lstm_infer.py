import cv2
import numpy as np
from lstm_cls import lstm_model, cnn_model

if __name__ == "__main__":
    # lstm model
    model = lstm_model(input_shape=(28,28))
    model.load_weights("weights/lstm_weights_05_acc_0.979_val_acc0.986.h5")

    # cnn model
    model = cnn_model(input_shape=(28,28,1))
    model.load_weights("weights/cnn_weights_05_acc_0.992_val_acc_1.000.h5")

    img_dir = "test/d0/d0_0063.png"

    cv_img = cv2.imread(img_dir, 0)
    tmp = cv_img / 255.
    tmp = np.expand_dims(tmp, axis=0)   # for lstm_model
    tmp = np.expand_dims(tmp,axis=-1)   # for cnn_model

    pred = model.predict(tmp)
    cls = np.argmax(pred)
    print(pred, cls)
