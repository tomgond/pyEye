import sys
import dlib
import pdb
import imutils
from imutils import face_utils
import random
import math
import psutil
import face_recognition
import trainer.models
from pprint import pprint
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, regularizers, GaussianDropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.applications import VGG16
import trainer.cv_classifiers as cv_classifiers
import trainer.models
import keras
import os

from trainer import models

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

MAX_DISTANCE_FROM_CENTER = 2202
# MAX_DISTANCE_FROM_CENTER = 1280

RESAMPLE_FACTOR = 7

VGG_CON_FEATURES_OUTPUT_SHAPE = (512,7,7)

import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face_predictor.dat")

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, index_file_path, images_folder_path, shuffle = True, predict_model = None, subset=None, resample_k=5, images_per_batch=4, output_shape = (3,224,224)):
        self.output_shape = output_shape
        self.counter = 0
        self.predict_model = predict_model
        self.shuffle = shuffle
        self.resample_k = resample_k
        self.images_per_batch = images_per_batch
        self.batch_size = images_per_batch
        'Initialization'
        self.images_folder_path = images_folder_path
        with open(index_file_path, 'r') as infile:
            lines = infile.readlines()
        t_list = map(lambda x: x.split(" "), lines)
        data_dict = {}
        self.total_images_with_aug = 0
        for itm in t_list:
            if os.path.exists(os.path.join(images_folder_path,itm[0])) and itm[0] in subset:
                x_val = int(itm[1].strip())
                y_val = int(itm[2].strip())
                full_im_path = os.path.join(self.images_folder_path, itm[0])
                img = cv2.imread(full_im_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                resample = number_of_samples_from_image_coors(x_val, y_val)
                data_dict[itm[0]] = {"lbl":np.array([x_val, y_val]), "resample": resample, "img_vector":x}
                self.total_images_with_aug += resample
        print("Generator created with {0} images, {1} images with augmentation".format(len(data_dict.keys()), self.total_images_with_aug))
        self.steps_per_epoch = self.total_images_with_aug // self.images_per_batch
        self.data = data_dict


    def image_name_to_K_predicted_VGG_outputs(self,img_name, k):
        train_datagen = ImageDataGenerator(
            width_shift_range=0.3,
            height_shift_range=0.3,
            data_format="channels_first",
            horizontal_flip=False,
            fill_mode='nearest')

        if self.predict_model != None:
            output = np.empty((k,) + VGG_CON_FEATURES_OUTPUT_SHAPE)
        else:
            output = np.empty((k, ) + (3,224,224))
        for i in range(0, k):
            self.counter += 1  # N generated
            if self.counter % 1000 == 0:
                print("Number of images in final test set:{0}".format(self.counter))
            aug_img = train_datagen.flow(self.data[img_name]["img_vector"], batch_size=self.resample_k).next()
            if self.predict_model != None:
                features = self.predict_model.predict(aug_img)
            else:
                features = aug_img
            features = features[0]
            output[i] = features
        return output

    def generate(self):
        'Generates batches of samples'
        # Infinite loop

        BiggerBuf = np.empty((self.batch_size+RESAMPLE_FACTOR,) + VGG_CON_FEATURES_OUTPUT_SHAPE)
        Ys = []
        imgs_counter = 0
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list(self.data.keys()) * 3)
            IDs = list(self.data.keys()) * 3
            # Generate batches
            j = 0
            for i in range(self.steps_per_epoch):
                # Filling buffer until there's enough image
                while imgs_counter < self.batch_size:
                    cur_image = self.data[IDs[indexes[j]]]
                    BiggerBuf[imgs_counter:imgs_counter+cur_image['resample']] = self.image_name_to_K_predicted_VGG_outputs(IDs[indexes[j]], cur_image['resample'])
                    Ys += [cur_image['lbl']] * cur_image['resample']
                    j += 1
                    imgs_counter += cur_image['resample']
                X = BiggerBuf[0:self.batch_size]
                y = Ys[0:self.batch_size]
                BiggerBuftmp = np.empty((self.batch_size + 20,) + VGG_CON_FEATURES_OUTPUT_SHAPE)
                if imgs_counter != self.batch_size:
                    BiggerBuftmp[0:imgs_counter-self.batch_size] = BiggerBuf[self.batch_size:imgs_counter]
                BiggerBuf = BiggerBuftmp
                Ys = Ys[self.batch_size:]
                imgs_counter -= self.batch_size

                # Generate data
                #X, y = self.__data_generation(list_IDs_temp)

                yield X, np.array(y)

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
        dist = math.sqrt(sum([pow(x -LEFT_SCREEN_AVG_X,2), pow(y - LEFT_SCREEN_AVG_Y,2)]))
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


def random_train_val_split(index_file_path,images_folder_path, test_ratio=0.8):
    with open(index_file_path, 'r') as infile:
        lines = infile.readlines()
    t_list = map(lambda x: x.split(" "), lines)
    train = []
    val = []
    for itm in t_list:
        if os.path.exists(os.path.join(images_folder_path, itm[0])):
            if random.random() > test_ratio:
                val.append(itm[0])
            else:
                train.append(itm[0])
    return train, val


def get_entire_face_from_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    rect = rects[0]
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    # image = image[rect[0]]
    # cv2.imshow("foobar", )
    # loop over the face parts individually
    tmp_imgs = []
    for (name, (i, j)) in list(
            filter(lambda x: x[0] in ["right_eye", "left_eye"], face_utils.FACIAL_LANDMARKS_IDXS.items())):
        print(name)
        # clone the original image so we can draw on it, then
        # display the name of the face part on the image
        clone = image.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        # loop over the subset of facial landmarks, drawing the
        # specific face part
        # for (x, y) in shape[i:j]:
        #     cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)


        # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=150, height=150, inter=cv2.INTER_CUBIC)
        one_eye = {"roi": roi, "clone": clone}
        tmp_imgs.append(one_eye)
        # show the particular face part
    for j, tmp_img in enumerate(tmp_imgs):
        cv2.imshow(str(j) + "1", tmp_img["roi"])
    cv2.imshow("all img", image)
    return tmp_imgs
    cv2.waitKey(0)

    # visualize all facial landmarks with a transparent overlay
    output = face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)


def transfer_learning(images_dir=None, index_path=None, resample_k=10, images_per_batch=4, batch_size=40):
    batch_size = resample_k * images_per_batch
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(3, 224, 224))
    base_model.summary()

    VGG_convolution_only = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
    VGG_convolution_only._make_predict_function()
    train_set, val_set = random_train_val_split(index_path, images_dir, test_ratio=0.8)
    print("[?] Train set size = {0}".format(len(train_set)))
    print("[?] eval set size = {0}".format(len(val_set)))

    train_gen = DataGenerator(index_path, images_dir, predict_model=VGG_convolution_only, subset=train_set, resample_k=resample_k, images_per_batch=images_per_batch)
    val_gen = DataGenerator(index_path, images_dir, predict_model=VGG_convolution_only, subset=val_set, resample_k=resample_k, images_per_batch=images_per_batch)


    my_model = models.transfer_v2()
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    rmsprop = keras.optimizers.rmsprop(lr=0.1, decay=0.01)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    my_model.compile(optimizer=adagrad, loss=euc_dist_keras, metrics=['accuracy'])

    epochs = 15
    print("Training model : \n"
          "Images per batch: {0}\n"
          "Train set images (including aug): {1}\n"
          "Val set images (including aug): {2}\n"
          "Train set steps: {3}\n"
          "Val set steps: {4}\n".format(
        images_per_batch,
        train_gen.total_images_with_aug,
        val_gen.total_images_with_aug,
        train_gen.steps_per_epoch,
        val_gen.steps_per_epoch))

    my_model.fit_generator(generator=train_gen.generate(),
                           steps_per_epoch=train_gen.steps_per_epoch ,
                           validation_data=val_gen.generate(),
                           validation_steps=val_gen.steps_per_epoch,
                           # callbacks=[LearningRateScheduler(lr_schedule),
                           #      ModelCheckpoint('model.h5', save_best_only=True)],
                           epochs=epochs)
    my_model.save("model.h5")

    os.system("gsutil -m cp model.h5 gs://pyeye_bucket/models/selu_bigger_model_more_samples_train.h5")
    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.


def all_me_model(images_dir="tmp", index_path="indexes.txt", resample_k=3, images_per_batch=10):

    train_set, val_set = random_train_val_split(index_path, images_dir, test_ratio=0.5)
    train_gen = DataGenerator(index_path, images_dir, predict_model=None, subset=train_set,
                              resample_k=resample_k, images_per_batch=images_per_batch)
    val_gen = DataGenerator(index_path, images_dir, predict_model=None, subset=val_set,
                            resample_k=resample_k, images_per_batch=images_per_batch)

    model = trainer.models.cnn_model_all_me(224)
    lr = 0.01
    adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    model.compile(loss=euc_dist_keras,
                  optimizer=adagrad,
                  metrics=['accuracy'])

    epochs = 15

    model.fit_generator(generator=train_gen.generate(),
                           steps_per_epoch=train_gen.steps_per_epoch ,
                           validation_data=val_gen.generate(),
                           validation_steps=val_gen.steps_per_epoch,
                           # callbacks=[LearningRateScheduler(lr_schedule),
                           #      ModelCheckpoint('model.h5', save_best_only=True)],
                           epochs=epochs)

    os.system("gsutil -m cp model.h5 gs://pyeye_bucket/models/only_me_no_transfer.h5")



if __name__ == "__main__":
    images_dir = "gs://pyeye_bucket/data/images_VGG16/"
    index_path = "gs://pyeye_bucket/data/indexes.txt"
    pir_prefix = "*"

    exit()
    imgs = os.listdir("tmp")
    for img in imgs:
        image = cv2.imread(os.path.join("tmp", img))
        get_entire_face_from_img(image)
    # copy_data_from_gs(images_dir=images_dir, index_path=index_path, pir_prefix=pir_prefix)
    # transfer_learning(images_dir="tmp", index_path="indexes.txt", resample_k=3, images_per_batch=32)
    all_me_model(images_dir="tmp", index_path="indexes.txt", resample_k=3, images_per_batch=5)
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
