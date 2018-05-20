import keras
from keras import Sequential, Input, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications import vgg16, VGG16
from keras.legacy.layers import merge

from trainer.cnn import VGG_CON_FEATURES_OUTPUT_SHAPE


def cnn_model_1(img_size):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='selu',
                     input_shape=(1,img_size,img_size)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='selu', padding='same'))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='selu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(250, activation='selu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    return model



def cnn_model_all_me(img_size):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='selu',
                     input_shape=(3,img_size,img_size)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='selu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='selu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='selu', padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1500, activation='selu'))
    model.add(Dense(970, activation='selu'))
    model.add(Dense(300, activation='selu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    return model


def cnn_model_3(img_size):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(1,img_size,img_size)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='selu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='tanh', padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='sigmoid', padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(250, activation='selu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    return model



def transfer_v2():
    my_model = keras.models.Sequential()
    my_model.add(Flatten(input_shape=VGG_CON_FEATURES_OUTPUT_SHAPE))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(2000, activation='selu'))
    my_model.add(Dense(1000, activation='selu'))
    my_model.add(Dense(500, activation='selu'))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(2))
    return my_model


def transfer_v1():
    my_model = keras.models.Sequential()
    my_model.add(Flatten(input_shape=VGG_CON_FEATURES_OUTPUT_SHAPE))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(1500, activation='selu'))
    my_model.add(Dense(970, activation='selu'))
    my_model.add(Dense(300, activation='selu'))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(2))
    return my_model
