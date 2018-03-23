from keras import Sequential, Input, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications import vgg16, VGG16
from keras.legacy.layers import merge





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



def cnn_model_2(img_size):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='selu',
                     input_shape=(1,img_size,img_size)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='selu', padding='same'))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='selu', padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(250, activation='selu'))
    model.add(Dense(150, activation='relu'))
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

