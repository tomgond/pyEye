import keras
from keras import Sequential, Input, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications import vgg16, VGG16
from keras.layers import Input, Embedding, LSTM, Dense, MaxPooling2D
from keras.models import Model


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

def cnn_model_multiple_inputs(face_shape, eye_shape):

    face_input = Input(shape=face_shape, name='face_input') #dtype='int32', name='main_input')
    left_eye_input = Input(shape=eye_shape, name='left_eye_input')
    right_eye_input = Input(shape=eye_shape, name='right_eye_input')

    face_x = (Conv2D(20, kernel_size=(2, 2),
                     activation='selu',
                     input_shape=face_input.shape))(face_input)
    face_x = (MaxPooling2D((3,3)))(face_x)
    face_x = (Conv2D(20, kernel_size=(2, 2), activation='relu', padding='same'))(face_x)
    face_x = (MaxPooling2D((3,3)))(face_x)
    face_x = (Conv2D(20, kernel_size=(2, 2), activation='selu', padding='same'))(face_x)
    face_x = (MaxPooling2D((3,3)))(face_x)
    face_x = Flatten()(face_x)
    face_x = Dropout(0.4)(face_x)
    face_x = Dense(2000,activation='relu')(face_x)
    face_x = BatchNormalization()(face_x)

    left_eye_x = (Conv2D(20, kernel_size=(2,2),
            activation='relu',
            input_shape=left_eye_input.shape))(left_eye_input)
    left_eye_x = (MaxPooling2D((3,3)))(left_eye_x)
    left_eye_x = (Conv2D(20, kernel_size=(2,2), activation='tanh', padding='same'))(left_eye_x)
    left_eye_x = (MaxPooling2D((3,3)))(left_eye_x)
    left_eye_x = (Conv2D(20, kernel_size=(2,2), activation='sigmoid', padding='same'))(left_eye_x)
    left_eye_x = (MaxPooling2D((3,3)))(left_eye_x)
    left_eye_x = Flatten()(left_eye_x)
    left_eye_x = Dropout(0.4)(left_eye_x)
    left_eye_x = Dense(2000,activation='selu')(left_eye_x)
    left_eye_x = BatchNormalization()(left_eye_x)

    right_eye_x = (Conv2D(20, kernel_size=(2,2),
            activation='relu',
            input_shape=right_eye_input.shape))(right_eye_input)
    right_eye_x = (MaxPooling2D((3,3)))(right_eye_x)
    right_eye_x = (Conv2D(20, kernel_size=(2,2), activation='selu', padding='same'))(right_eye_x)
    right_eye_x = (MaxPooling2D((3,3)))(right_eye_x)
    right_eye_x = (Conv2D(20, kernel_size=(2,2), activation='relu', padding='same'))(right_eye_x)
    right_eye_x = (MaxPooling2D((3,3)))(right_eye_x)
    right_eye_x = Flatten()(right_eye_x)
    right_eye_x = Dropout(0.2)(right_eye_x)
    right_eye_x = Dense(2000, activation='relu')(right_eye_x)
    right_eye_x = BatchNormalization()(right_eye_x)

    joint_layer = keras.layers.Concatenate()([left_eye_x, face_x, right_eye_x])
    # joint_layer = keras.layers.concatenate([left_eye_x, face_x, right_eye_x])
    joint_layer = Dense(2000, activation='selu')(joint_layer)
    joint_layer = Dropout(0.5)(joint_layer)
    joint_layer = Dense(1000, activation='tanh')(joint_layer)
    joint_layer = Dropout(0.5)(joint_layer)
    joint_layer = Dense(500, activation='sigmoid')(joint_layer)
    joint_layer = Dropout(0.5)(joint_layer)
    joint_layer = Dense(250, activation='sigmoid')(joint_layer)
    joint_layer = Dropout(0.5)(joint_layer)
    joint_layer = Dense(100, activation='sigmoid')(joint_layer)
    model_output = Dense(2)(joint_layer)
    return Model(inputs=[face_input, left_eye_input, right_eye_input], outputs=model_output)
