from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split



def my_CNN(x, y, classes):
    batch = 32
    classes = len(classes)
    epoch = 20
    filters = 128
    pools = 2
    convolutions = 3

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

    uniques, id_train = np.unique(y_train, return_inverse=True)
    y_train = np_utils.to_categorical(id_train, classes)
    uniques, id_test = np.unique(y_test, return_inverse=True)
    y_test = np_utils.to_categorical(id_test, classes)


    model = Sequential()
    model.add(Convolution2D(filters, convolutions, convolutions, border_mode='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(int(filters/2), convolutions, convolutions, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(int(filters/4), convolutions, convolutions, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(int(filters/8), convolutions, convolutions, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pools, pools)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return x_train, y_train, x_test, y_test, model


