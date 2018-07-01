import keras
import math
from keras.optimizers import *
from keras import Sequential, Input, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications import vgg16, VGG16
from keras.layers import Input, Embedding, LSTM, Dense, MaxPooling2D
from keras.models import Model


N_CHANNELS = 3

FACE_IMAGE_SIZE_X = 250
FACE_IMAGE_SIZE_Y = 250

EYE_IMAGE_SIZE_X = 100
EYE_IMAGE_SIZE_Y = 150

LEFT_SCREEN_AVG_X = 798
LEFT_SCREEN_AVG_Y = 500

def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

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

KERNEL_SIZE = 2
MAX_POOL_SIZE = 3

'''
    Input dict features:
    
    left_eye_n_layers
    face_n_layers
    right_eye_n_layers
    max_pooling_2d_size
    conv_kernel_size
    conv_init_size
    conv_factor
    optimizer
    n_finish_layers
    finish_layers_factor
    
    N finish layers
    finish layers factor
    



'''


def model_generator(input_dict):
    pass


def cnn_model_multiple_inputs(face_shape=(N_CHANNELS,FACE_IMAGE_SIZE_X, FACE_IMAGE_SIZE_Y), eye_shape=(N_CHANNELS, EYE_IMAGE_SIZE_X, EYE_IMAGE_SIZE_Y), params=None):
    '''
    
    :param params:
        conv_funcs (int)
        kernel_size (2,2)
        max_pooling (2,2)
        optimizer = adam, adadelta, ...
        dropout = 0.1 0.2 0.3
        initial_final_layer = number of neurons initally 
        final_layers = number of layers in the end
        final_layer_decrease_factor
    :return: 
    '''
    face_input = Input(shape=face_shape, name='face_input') #dtype='int32', name='main_input')
    left_eye_input = Input(shape=eye_shape, name='left_eye_input')
    right_eye_input = Input(shape=eye_shape, name='right_eye_input')

    face_x = (Conv2D(params['conv_funcs'], kernel_size=params['kernel_size'],
                     activation='selu',
                     input_shape=face_input.shape))(face_input)
    face_x = (MaxPooling2D(params['max_pooling']))(face_x)
    face_x = (Conv2D(params['conv_funcs'], kernel_size=params['kernel_size'], activation='relu', padding='same'))(face_x)
    face_x = (MaxPooling2D(params['max_pooling']))(face_x)
    face_x = (Conv2D(params['conv_funcs'], kernel_size=params['kernel_size'], activation='selu', padding='same'))(face_x)
    face_x = (MaxPooling2D(params['max_pooling']))(face_x)
    face_x = Flatten()(face_x)
    face_x = Dropout(params['dropout'])(face_x)
    face_x = Dense(2000,activation='relu')(face_x)
    face_x = BatchNormalization()(face_x)

    left_eye_x = (Conv2D(params['conv_funcs'], kernel_size=params['kernel_size'],
            activation='relu',
            input_shape=left_eye_input.shape))(left_eye_input)
    left_eye_x = (MaxPooling2D(params['max_pooling']))(left_eye_x)
    left_eye_x = (Conv2D(params['conv_funcs'], kernel_size=params['kernel_size'], activation='tanh', padding='same'))(left_eye_x)
    left_eye_x = (MaxPooling2D(params['max_pooling']))(left_eye_x)
    left_eye_x = (Conv2D(params['conv_funcs'], kernel_size=params['kernel_size'], activation='sigmoid', padding='same'))(left_eye_x)
    left_eye_x = (MaxPooling2D(params['max_pooling']))(left_eye_x)
    left_eye_x = Flatten()(left_eye_x)
    left_eye_x = Dropout(params['dropout'])(left_eye_x)
    left_eye_x = Dense(2000,activation='selu')(left_eye_x)
    left_eye_x = BatchNormalization()(left_eye_x)

    right_eye_x = (Conv2D(params['conv_funcs'], kernel_size=params['kernel_size'],
            activation='relu',
            input_shape=right_eye_input.shape))(right_eye_input)
    right_eye_x = (MaxPooling2D(params['max_pooling']))(right_eye_x)
    right_eye_x = (Conv2D(params['conv_funcs'], kernel_size=params['kernel_size'], activation='selu', padding='same'))(right_eye_x)
    right_eye_x = (MaxPooling2D(params['max_pooling']))(right_eye_x)
    right_eye_x = (Conv2D(params['conv_funcs'], kernel_size=params['kernel_size'], activation='relu', padding='same'))(right_eye_x)
    right_eye_x = (MaxPooling2D(params['max_pooling']))(right_eye_x)
    right_eye_x = Flatten()(right_eye_x)
    right_eye_x = Dropout(params['dropout'])(right_eye_x)
    right_eye_x = Dense(2000, activation='relu')(right_eye_x)
    right_eye_x = BatchNormalization()(right_eye_x)

    joint_layer = keras.layers.Concatenate()([left_eye_x, face_x, right_eye_x])
    # joint_layer = keras.layers.concatenate([left_eye_x, face_x, right_eye_x])
    n_neurons = params['initial_final_layer']
    joint_layer = Dense(n_neurons, activation='selu')(joint_layer)
    joint_layer = Dropout(params['dropout'])(joint_layer)
    for layer_in in range(0,params['final_layers']):
        n_neurons = int(math.ceil(n_neurons*params['final_layer_decrease_factor']))
        joint_layer = Dense(n_neurons, activation='tanh')(joint_layer)
        joint_layer = Dropout(params['dropout'])(joint_layer)
    model_output = Dense(2)(joint_layer)

    mymodel = Model(inputs=[face_input, left_eye_input, right_eye_input], outputs=model_output)
    mymodel.compile(loss=euc_dist_keras,
                  optimizer=keras.optimizers.Adadelta(lr=float(params['lr'])),
                  metrics=['accuracy'])

    return mymodel
