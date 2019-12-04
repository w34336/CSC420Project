from keras.preprocessing.image import  img_to_array
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import dataProcess
import myCNN



def train(x_train, y_train, x_test, y_test, model):
    nb_epoch = 60
    batch_size = 32
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(x_test, y_test))
    model.save("weapon_detection2.h5", overwrite=True)

def test(test_path):
    width, height = 240, 240
    model = tf.keras.models.load_model("weapon_detection2.h5")
    files = os.listdir(test_path)

    for i in files:
        if i.startswith('.'):
            continue
        im = Image.open(test_path + '/' + i)
        imrs = im.resize((width, height))
        imrs = img_to_array(imrs)/255
        imrs = imrs.transpose(2, 0, 1)
        imrs = imrs.reshape(3, width, height)
        imrs = np.array([imrs])
        predictions = model.predict(imrs)
        print(i)
        print(predictions)

if __name__ == '__main__':
    path1 = './test'
    # path2 = './train'
    # x, y, classes = dataProcess.load_images(path2)
    # x_train, y_train, x_test, y_test, model = myCNN.my_CNN(x, y, classes)
    # train(x_train, y_train, x_test, y_test, model)
    test(path1)