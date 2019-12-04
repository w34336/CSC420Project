from keras.preprocessing.image import img_to_array
import numpy as np
import os
from PIL import Image

def load_images(train_path):
    width, height = 240, 240
    classes = os.listdir(train_path)
    for category in classes:
        if category.startswith('.'):
            classes.remove(category)
    x = []
    y = []
    count = 0
    for fol in classes:
        imgfiles = os.listdir(train_path + '/' + fol)
        for img in imgfiles:
            if img.startswith('.'):
                continue
            im = Image.open(train_path + '/' + fol + '/' + img)
            im = im.convert(mode='RGB')
            imrs = im.resize((width, height))
            imrs = img_to_array(imrs) / 255
            imrs = imrs.transpose(2, 0, 1)
            imrs = imrs.reshape(3, width, height)
            x.append(imrs)
            y.append(count)
        count += 1

    x = np.array(x)
    y = np.array(y)
    # print(x)
    # print(y)
    # print(classes)
    return x, y, classes