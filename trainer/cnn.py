import sys
import pdb
import random
import math
import psutil
from pprint import pprint
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, regularizers, GaussianDropout
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
from keras import backend as K, Model, optimizers

K.set_image_data_format('channels_first')

CROP_SIZE = 250
IMG_SIZE = 150


LEFT_SCREEN_AVG_X = 798
LEFT_SCREEN_AVG_Y = 500

RIGHT_SCREEN_AVG_X = -1150
RIGHT_SCREEN_AVG_Y = 455

MAX_DISTANCE_FROM_CENTER = 1280

RESAMPLE_FACTOR  = 4

VGG_CON_FEATURES_OUTPUT_SHAPE = (512,7,7)

import numpy as np



class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, index_file_path, images_folder_path, shuffle = True, predict_model = None, subset=None, resample_k=5, images_per_batch=4):
        self.counter = 0
        self.predict_model = predict_model
        self.shuffle = shuffle
        self.resample_k = resample_k
        self.images_per_batch = images_per_batch
        self.batch_size = resample_k * images_per_batch
        'Initialization'
        self.images_folder_path = images_folder_path
        with open(index_file_path, 'r') as infile:
            lines = infile.readlines()
        t_list = map(lambda x: x.split(" "), lines)
        data_dict = {}
        for itm in t_list:
            if os.path.exists(os.path.join(images_folder_path,itm[0])) and itm[0] in subset:
                x_val = int(itm[1].strip())
                y_val = int(itm[2].strip())
                full_im_path = os.path.join(self.images_folder_path, itm[0])
                img = cv2.imread(full_im_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                data_dict[itm[0]] = {"lbl":np.array([x_val, y_val]), "resample": number_of_samples_from_image_coors(x_val,y_val), "img_vector":x}

        print("Generator created with {0} images".format(len(data_dict.keys())))
        self.data = data_dict


    def image_name_to_K_predicted_VGG_outputs(self,img_name, k):
        train_datagen = ImageDataGenerator(
            width_shift_range=0.3,
            height_shift_range=0.3,
            data_format="channels_first",
            horizontal_flip=False,
            fill_mode='nearest')

        output = np.empty((k,) + VGG_CON_FEATURES_OUTPUT_SHAPE)
        for i in range(0, k):
            self.counter += 1  # N generated
            if self.counter % 1000 == 0:
                print("Number of images in final test set:{0}".format(self.counter))
            aug_img = train_datagen.flow(self.data[img_name]["img_vector"], batch_size=self.resample_k).next()
            features = self.predict_model.predict(aug_img)
            features = features[0]
            output[i] = features
        return output

    def generate(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(self.data.keys())
            IDs = list(self.data.keys())
            # Generate batches
            imax = int(math.floor((len(indexes))/self.images_per_batch))
            for i in range(imax):
                # Find list of IDs
                # pdb.set_trace()
                list_IDs_temp = [IDs[k] for k in indexes[int(i*self.images_per_batch):(i+1)*self.images_per_batch]]

                # Generate data
                X, y = self.__data_generation(list_IDs_temp)

                yield X, y

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, images_list, use_rescale_weight=False):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization

        Xs = np.empty((self.batch_size,) + VGG_CON_FEATURES_OUTPUT_SHAPE)
        Ys = []
        # Generate data
        for i, ID in enumerate(images_list):
            # Store volume
            Xs[i*self.resample_k:(i*self.resample_k)+self.resample_k] = self.image_name_to_K_predicted_VGG_outputs(ID, self.resample_k)
            Ys += [self.data[ID]['lbl']]*self.resample_k
        #
        # while (len(Xs) > self.batch_size):
        #     r = random.randint(0,len(Xs))
        #     Xs.pop(r)
        #     Ys.pop(r)

        return Xs, np.array(Ys)


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


def random_train_val_split(index_file_path,images_folder_path, test_ratio = 0.8):
    with open(index_file_path, 'r') as infile:
        lines = infile.readlines()
    t_list = map(lambda x: x.split(" "), lines)
    train = []
    val = []
    for itm in t_list:
        if os.path.exists(os.path.join(images_folder_path, itm[0])):
            if random.random()>test_ratio:
                val.append(itm[0])
            else:
                train.append(itm[0])
    return train, val

def get_my_prediction_model():
    my_model = keras.models.Sequential()
    my_model.add(Flatten(input_shape=VGG_CON_FEATURES_OUTPUT_SHAPE))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(1500, activation='relu'))
    my_model.add(Dense(970, activation='relu'))
    my_model.add(Dense(300, activation='relu'))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(2))
    return my_model


def transfer_learning(images_dir=None, index_path=None, resample_k=10, images_per_batch=4):
    batch_size = resample_k * images_per_batch
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(3, 224, 224))
    base_model.summary()

    VGG_convolution_only = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
    VGG_convolution_only._make_predict_function()
    train_set, val_set = random_train_val_split(index_path, images_dir, test_ratio=0.7)
    print("[?] Train set size = {0}".format(len(train_set)))
    print("[?] eval set size = {0}".format(len(val_set)))
    train_gen = DataGenerator(index_path, images_dir, predict_model=VGG_convolution_only, subset=train_set, resample_k=resample_k, images_per_batch=images_per_batch).generate()
    val_gen = DataGenerator(index_path, images_dir, predict_model=VGG_convolution_only, subset=val_set, resample_k=resample_k, images_per_batch=images_per_batch).generate()


    my_model = get_my_prediction_model()
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    rmsprop = keras.optimizers.rmsprop(lr=0.1, decay=0.01)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    my_model.compile(optimizer=adagrad, loss=euc_dist_keras, metrics=['accuracy'])

    epochs = 30
    print("Training model : \n Steps per epoch : {0} \n epochs: {1}".format((len(train_set) // images_per_batch), epochs))

    my_model.fit_generator(generator=train_gen,
                           steps_per_epoch=len(train_set)*resample_k // batch_size,
                           validation_data=val_gen,
                           validation_steps=len(val_set)*resample_k // batch_size,
                           # callbacks=[LearningRateScheduler(lr_schedule),
                           #      ModelCheckpoint('model.h5', save_best_only=True)],
                           epochs=epochs)
    my_model.save("model.h5")

    os.system("gsutil -m cp model.h5 gs://pyeye_bucket/models/this_is_kickasss.h5")
    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.



if __name__ == "__main__":
    images_dir = "gs://pyeye_bucket/data/images_VGG16/"
    index_path = "gs://pyeye_bucket/data/indexes.txt"
    pir_prefix = "*"
    copy_data_from_gs(images_dir=images_dir, index_path=index_path, pir_prefix=pir_prefix)
    transfer_learning(images_dir="tmp", index_path="indexes.txt", resample_k=3, images_per_batch=20)
    exit(0)


    # X,y = load_data()
    # model = trainer.models.cnn_model_2(IMG_SIZE)
    # lr = 0.01
    # rmsprop = keras.optimizers.rmsprop(lr=0.01,decay=0.01)
    # sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.99, decay=0.001)
    # adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    # adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    # model.compile(loss=euc_dist_keras,
    #               optimizer=adam,
    #               metrics=['accuracy'])
    #
    # batch_size = 40
    # epochs = 20
    #
    # model.fit(X, y,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_split=0.2,
    #           callbacks=[LearningRateScheduler(lr_schedule),
    #                      ModelCheckpoint('model.h5', save_best_only=True)]
    #           )
    #
    # os.system("gsutil -m cp model.h5 gs://pyeye_bucket/models/model3_adam.h5")
