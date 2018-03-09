import sys
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import cv_classifiers
import random
import keras
import Image
import matplotlib.pyplot as plt
import os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
from keras.layers.normalization import BatchNormalization
import os

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-8.0/lib64'
from skimage import color, exposure, transform
from skimage import io
import os
import glob
from utils import constants
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import keras.callbacks as cbks
from keras.utils import plot_model
import cv2
import time
import os
from keras import backend as K
K.set_image_data_format('channels_first')

IMG_SIZE = 300


def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(1,IMG_SIZE,IMG_SIZE)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    return model

def crop_many_sizes(img):
    y = random.randint(0, 340)
    x = random.randint(0, 180)
    crop_image = img[x:x+300, y:y+300]
    return crop_image

def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    cv2.namedWindow("after", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("after", img)
    cv2.waitKey(10)
    return img





def load_data():
    train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        data_format="channels_first",
        horizontal_flip=False,
        fill_mode='nearest')
    counter = 0
    imgs = []
    lbls = []
    with open(constants.TRAIN_INDEX_PATH, "r") as indexes_file:
        for img_path, x_cor, y_cor in map(lambda x: x.split(), indexes_file.readlines()):
            print img_path
            try:
                full_im_path = os.path.join(constants.TRAIN_IMAGES_PATH,img_path)
                img = cv2.imread(full_im_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv_classifiers.crop_face(img,IMG_SIZE)
                img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                i = 0
                # imgs = train_datagen.flow(x, batch_size=1, save_to_dir='augment', save_prefix='{0}_{1}'.format(x_cor, y_cor), save_format='jpeg')
                for _ in range(0,15):
                    counter += 1
                    if counter % 100 == 0:
                        print counter
                    aug_img = train_datagen.flow(x).next()
                    aug_img = aug_img.reshape((1,300,300))
                    imgs.append(aug_img)
                    lbls.append((int(x_cor),int(y_cor)))
            except Exception as e:
                print e
                continue


    X = np.array(imgs, dtype='float32')
    # Make one hot targets
    Y = np.array(lbls)
    return X, Y

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))



if __name__ == "__main__":


    X,y = load_data()

    model = cnn_model()
    lr = 0.01
    rmsprop = keras.optimizers.rmsprop(lr=0.01,decay=0.01)
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.99, decay=0.001)
    model.compile(loss=euc_dist_keras,
                  optimizer=adam,
                  metrics=['accuracy'])

    batch_size = 32
    epochs = 40

    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[LearningRateScheduler(lr_schedule),
                         ModelCheckpoint('model.h5', save_best_only=True)]
              )


    print X
    print y
