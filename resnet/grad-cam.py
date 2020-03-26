import keras
import cv2
import numpy as np
import keras.backend as K
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img,img_to_array
K.set_learning_phase(1)   # set learning phase
from resnet_cls import PowerTransferMode


transfer = PowerTransferMode()
model = transfer.resnet18_model(lr=3e-4, decay=1e-6, nb_classes=2, img_size=28, target_size=28, RGB=False)
model.load_weights("../weights/r18_weights_05_val_acc_0.969.h5")

test_img = cv2.imread("../test/d0/d0_0063.png", 0)
x = np.reshape(test_img, (1,28,28,1))

y = model.predict(x)
class_idx = np.argmax(y[0])
print(class_idx)


last_conv_layer = model.get_layer("add_8")      # (,2,2,64)
class_output = model.output[:,class_idx]    # (,1)
grads = K.gradients(class_output,last_conv_layer.output)[0]    # (,2,2,64)
avgconv_grads = K.mean(grads,axis=(0,1,2))
func = K.function([model.input],[avgconv_grads,last_conv_layer.output[0]])
avg_conv_grads_value, conv_layer_output_value = func([x])
for i in range(64):
    conv_layer_output_value[:,:,i] *= avg_conv_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap,0)
heatmap /= np.max(heatmap)

heatmap = cv2.resize(heatmap,(28,28))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
added_img = cv2.addWeighted(cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR),0.6,heatmap,0.4,0)
cv2.imshow('grad-cam',heatmap)
cv2.waitKey(0)
cv2.imshow('grad-cam',added_img)
cv2.waitKey(0)
