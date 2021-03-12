from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.datasets import mnist, cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

CLIP_MIN = -0.5
CLIP_MAX = 0.5
PATH_DATA = "./data/"

def get_data(dataset='cifar10-cifar100', noiserate=0.1):
    if dataset == 'cifar10-cifar100':
        (X_train0, y_train0), (X_test, y_test) = cifar10.load_data()
        (X_train1, y_train1), (X_test1, y_test1) = cifar100.load_data()

        np.random.seed(0)
        X_train_openset = X_train1[np.random.choice(X_train1.shape[0], int(X_train0.shape[0]*noiserate), replace=False), :]
        y_train_openset = np.repeat(range(10), X_train0.shape[0]*noiserate/10).reshape(-1,1)

        X_train = np.concatenate((X_train0[0:int(X_train0.shape[0]*(1-noiserate))], X_train_openset), axis=0)
        y_train = np.concatenate((y_train0[0:int(X_train0.shape[0]*(1-noiserate))], y_train_openset), axis=0)


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = (X_train/255.0) - (1.0 - CLIP_MAX)
    X_test = (X_test/255.0) - (1.0 - CLIP_MAX)

    np.save('./data/X_train.npy', X_train)
    np.save('./data/y_train', y_train)
    np.save('./data/X_test', X_test)
    np.save('./data/y_test', y_test)

    return X_train, y_train, X_test, y_test

def get_model(dataset='cifar10'):
    if dataset == 'cifar10':
        layers = [
            Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)),  # 0
            BatchNormalization(),
            Activation('relu'),  # 1
            Conv2D(64, (3, 3), padding='same'),  # 2
            BatchNormalization(),
            Activation('relu'),  # 3
            MaxPooling2D(pool_size=(2, 2)),  # 4
            Conv2D(128, (3, 3), padding='same'),  # 5
            BatchNormalization(),
            Activation('relu'),  # 6
            Conv2D(128, (3, 3), padding='same'),  # 7
            BatchNormalization(),
            Activation('relu'),  # 8
            MaxPooling2D(pool_size=(2, 2)),  # 9
            Conv2D(196, (3, 3), padding='same'),  # 10
            BatchNormalization(),
            Activation('relu'),  # 11
            Conv2D(196, (3, 3), padding='same'),  # 12
            BatchNormalization(),
            Activation('relu'),  # 13
            MaxPooling2D(pool_size=(2, 2)),  # 14
            Flatten(),  # 15
            Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        ]

    model = Sequential()
    for layer in layers:
        model.add(layer)

    return model
