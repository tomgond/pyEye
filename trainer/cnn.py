import sys
import math
import psutil

from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.applications import VGG16
import trainer.cv_classifiers as cv_classifiers
import trainer.models
import keras
import os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import cv2
import os
import gc
from keras import backend as K, Model

K.set_image_data_format('channels_first')

CROP_SIZE = 250
IMG_SIZE = 150


LEFT_SCREEN_AVG_X = 798
LEFT_SCREEN_AVG_Y = 500

RIGHT_SCREEN_AVG_X = -1150
RIGHT_SCREEN_AVG_Y = 455

MAX_DISTANCE_FROM_CENTER = 1280

RESAMPLE_FACTOR  = 4

def print_memory_statistics():
    process = psutil.Process(os.getpid())
    print("======= Memory info ==============")
    print("Memory percent: {0}".format(process.memory_percent()))
    print("Full memory info : {0}".format(process.memory_full_info()))



def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))


def copy_data_from_gs(images_dir="gs://pyeye_bucket/data/images_VGG16/", pir_prefix="132*", index_path="gs://pyeye_bucket/data/indexes.txt" , to_save_dir='tmp/'):
    os.mkdir("tmp")
    print("[ ] Running gsutil command")
    os.system("gsutil -m cp {0}{1} {2}".format(images_dir,pir_prefix, to_save_dir))
    print("[ ] Finished, getting indexes")
    os.system("gsutil cp {0} indexes.txt".format(index_path))
    print("[ ] Getting haar cascades")
    os.mkdir("haar")
    os.system("gsutil cp gs://pyeye_bucket/haar_cascades/* haar/")



def img_reprocess(img, crop_size=CROP_SIZE, img_size=IMG_SIZE, to_greyscale=True):
    if len(img.shape) != 3:
        "[X] Did not read anything from cv2.imread"
    if to_greyscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(img.shape) > 2:
        "[X] Did not read anything from cv2.cvtColor"
    img = cv_classifiers.crop_face(img, crop_size)
    img = cv2.resize(img, (img_size, img_size))
    return img


def number_of_samples_from_image_coors(x, y):
    if x > 0:
        dist = math.sqrt(sum([pow(x-LEFT_SCREEN_AVG_X,2), pow(y-LEFT_SCREEN_AVG_Y,2)]))
    else:
        dist = math.sqrt(sum([pow(x - RIGHT_SCREEN_AVG_X,2), pow(y - RIGHT_SCREEN_AVG_Y,2)]))
    return int(math.ceil((float(dist)/MAX_DISTANCE_FROM_CENTER)*RESAMPLE_FACTOR))



def load_data_ver2(images_base_path="tmp", grayscale=False, preprocessing_model=None):
    print("[ ] Entering load data.")
    print_memory_statistics()
    train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        data_format="channels_first",
        horizontal_flip=False,
        fill_mode='nearest',
        preprocessing_function=lambda x: preprocessing_model.predict(x))
    imgs = []
    lbls = []
    with open("indexes.txt", "r") as indexes_file:
        i = 0
        counter=0
        for img_path, x_cor, y_cor in map(lambda x: x.split(), indexes_file.readlines()):
            i += 1
            if i % 1000 == 0:
                gc.collect()
                print_memory_statistics()
                print("[ ] Images in X : {0}".format(i))
            # print(img_path)
            try:
                full_im_path = os.path.join(images_base_path,img_path)
                # print("[ ] Loading file : {0}".format(full_im_path))
                if not(os.path.exists(full_im_path)):
                    continue
                n_resample = number_of_samples_from_image_coors(int(x_cor),int(y_cor))
                #print("Sampling with : {0}".format(n_resample))
                if grayscale:
                    img = cv2.imread(full_im_path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(full_im_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                for _ in range(0,n_resample):
                    counter += 1
                    if counter % 1000 == 0:
                        print("Number of images in final test set:{0}".format(counter))
                    aug_img = train_datagen.flow(x).next()
                    aug_img = aug_img[0]
                    imgs.append(aug_img)
                    lbls.append((int(x_cor),int(y_cor)))
            except Exception as e:
                print(e)
                continue

    X = np.array(imgs, dtype='float32')
    # Make one hot targets
    Y = np.array(lbls)
    return X, Y





def load_data(images_base_path="tmp", grayscale=False):
    print("[ ] Entering load data.")
    print_memory_statistics()
    train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        data_format="channels_first",
        horizontal_flip=False,
        fill_mode='nearest')
    imgs = []
    lbls = []
    with open("indexes.txt", "r") as indexes_file:
        i = 0
        counter=0
        for img_path, x_cor, y_cor in map(lambda x: x.split(), indexes_file.readlines()):
            i += 1
            if i % 1000 == 0:
                gc.collect()
                print_memory_statistics()
                print("[ ] Images in X : {0}".format(i))
            # print(img_path)
            try:
                full_im_path = os.path.join(images_base_path,img_path)
                # print("[ ] Loading file : {0}".format(full_im_path))
                if not(os.path.exists(full_im_path)):
                    continue
                n_resample = number_of_samples_from_image_coors(int(x_cor),int(y_cor))
                #print("Sampling with : {0}".format(n_resample))
                if grayscale:
                    img = cv2.imread(full_im_path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(full_im_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                for _ in range(0,n_resample):
                    counter += 1
                    if counter % 1000 == 0:
                        print("Number of images in final test set:{0}".format(counter))
                    aug_img = train_datagen.flow(x).next()
                    aug_img = aug_img[0]
                    imgs.append(aug_img)
                    lbls.append((int(x_cor),int(y_cor)))
            except Exception as e:
                print(e)
                continue

    X = np.array(imgs, dtype='float32')
    # Make one hot targets
    Y = np.array(lbls)
    return X, Y




def lr_schedule(epoch):
    lr = 0.01
    return lr * (0.1 ** int(epoch / 10))


def transfer_learning(model_name, last_layer):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(3, 224, 224))
    base_model.summary()

    VGG_convolution_only = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)

    X, y = load_data(images_base_path="tmp")
    features = []
    i = 0
    print("Shape of input X matrix:")
    print(X.shape)
    for x_img, coors in zip(X,y):
        if i%1000 == 0:
            print("We're at : {0}".format(i))
        x_img = np.expand_dims(x_img, axis=0)
        features.append(VGG_convolution_only.predict(x_img))
        i +=1
    np_features = np.array(features)
    np_features = np_features.squeeze(1)
    print("Saving features")
    np.save("np_features.np", np_features)
    print("Features saved")
    print("Features size : {0}".format(os.path.getsize("np_features.np.npy")))
    my_model = keras.models.Sequential()
    my_model.add(Flatten(input_shape=features[0].shape))
    my_model.add(Dense(1000, activation='relu'))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(4000, activation='relu'))
    my_model.add(Dense(2))
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.99, decay=0.001)
    my_model.compile(optimizer=adam, loss=euc_dist_keras, metrics=['accuracy'])
    batch_size = 40
    epochs = 30

    my_model.fit(np.array(features), y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[LearningRateScheduler(lr_schedule),
                         ModelCheckpoint('model.h5', save_best_only=True)]
              )

    os.system("gsutil -m cp model.h5 gs://pyeye_bucket/models/the_madman_has_done_it.h5")
    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.



if __name__ == "__main__":
    copy_data_from_gs(images_dir="gs://pyeye_bucket/data/images_VGG16/", index_path="gs://pyeye_bucket/data/indexes.txt", pir_prefix="*")
    transfer_learning('VGG16', 'fc2')
    exit(0)


    X,y = load_data()
    model = trainer.models.cnn_model_2(IMG_SIZE)
    lr = 0.01
    rmsprop = keras.optimizers.rmsprop(lr=0.01,decay=0.01)
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.99, decay=0.001)
    adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model.compile(loss=euc_dist_keras,
                  optimizer=adam,
                  metrics=['accuracy'])

    batch_size = 40
    epochs = 20

    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[LearningRateScheduler(lr_schedule),
                         ModelCheckpoint('model.h5', save_best_only=True)]
              )
    
    os.system("gsutil -m cp model.h5 gs://pyeye_bucket/models/model3_adam.h5")
