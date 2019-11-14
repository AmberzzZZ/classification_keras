import cv2
import numpy as np
from resnet_cls import PowerTransferMode


if __name__ == "__main__":

    image_size = 512
    target_size = 64
    n_classes = 2

    transfer = PowerTransferMode()

    # resnet18
    model = transfer.resnet18_model(lr=3e-4, decay=1e-6, nb_classes=2,
                                    img_size=image_size, target_size=target_size, RGB=False)
    model.load_weights("weights/r18_weights_05_val_acc_0.969.h5")

    img_dir = "test/d0/d0_0063.png"

    cv_img = cv2.imread(img_dir, 0)
    tmp = cv_img / 255.
    tmp = np.expand_dims(tmp, axis=0)   # for lstm_model
    tmp = np.expand_dims(tmp,axis=-1)   # for cnn_model

    pred = model.predict(tmp)
    cls = np.argmax(pred)
    print(pred, cls)


