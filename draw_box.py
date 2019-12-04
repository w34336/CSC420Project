import numpy as np
import tensorflow as tf
import cv2
import os
from keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt

def calculate_boundingbox(img, model, n, classes):
    img = img_to_array(img)
    img  = cv2.resize(img, (2400, 2400))
    img = img / 255
    width, height = img.shape[0], img.shape[1]
    for i in range(0, width, n):
        for j in range(0, height, n):
            patch = img[i:i+n, j:j+n]
            plt.imshow(patch)
            plt.show()
            patch = patch.transpose(2, 0, 1)
            patch = patch.reshape(3, 240, 240)
            patch = np.array([patch])
            print(model.predict(patch))
            print(model.predict(patch)[0][0])
            if classes == 'gun' and model.predict(patch)[0][0] > 0.9:
                img[i:i+n, j:j+n] = 255
    img = img * 255
    cv2.imwrite('./test/result.jpg', img)

if __name__ == '__main__':
    path = "./test"
    classes = 'gun'
    img = Image.open(path+'/gun2.jpeg')
    model = tf.keras.models.load_model("weapon_detection2.h5")
    calculate_boundingbox(img, model, 240, classes)


