import sys
import math
from keras.preprocessing.image import ImageDataGenerator, img_to_array
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
from keras import backend as K


K.set_image_data_format('channels_first')

CROP_SIZE = 250
IMG_SIZE = 150


LEFT_SCREEN_AVG_X = 798
LEFT_SCREEN_AVG_Y = 500

RIGHT_SCREEN_AVG_X = -1150
RIGHT_SCREEN_AVG_Y = 455

MAX_DISTANCE_FROM_CENTER = 1280

RESAMPLE_FACTOR  = 10


def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))



def copy_data_from_gs():
    os.mkdir("tmp")
    print("[ ] Running gsutil command")
    os.system("gsutil -m cp gs://pyeye_bucket/data/images/* tmp/")
    print("[ ] Finished, getting indexes")
    os.system("gsutil cp gs://pyeye_bucket/data/indexes.txt indexes.txt")
    print("[ ] Getting haar cascades")
    os.mkdir("haar")
    os.system("gsutil cp gs://pyeye_bucket/haar_cascades/* haar/")


def img_reprocess(img, crop_size=CROP_SIZE, img_size=IMG_SIZE):
    if len(img.shape) != 3:
        "[X] Did not read anything from cv2.imread"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(img.shape) != 3:
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
    with open("indexes.txt", "r") as indexes_file:
        for img_path, x_cor, y_cor in map(lambda x: x.split(), indexes_file.readlines()):
            # print(img_path)
            try:
                full_im_path = os.path.join("tmp",img_path)
                # print("[ ] Loading file : {0}".format(full_im_path))
                if not(os.path.exists(full_im_path)):
                    continue
                print("[V] File exists... we continue")
                n_resample = number_of_samples_from_image_coors(int(x_cor),int(y_cor))
                print("Sampling with : {0}".format(n_resample))
                img = cv2.imread(full_im_path, cv2.IMREAD_GRAYSCALE)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                print("[ ] Xs shape: {0}".format(x.shape))
                i = 0
                # imgs = train_datagen.flow(x, batch_size=1, save_to_dir='augment', save_prefix='{0}_{1}'.format(x_cor, y_cor), save_format='jpeg')
                for _ in range(0,n_resample):
                    counter += 1
                    if counter % 100 == 0:
                        print(counter)
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
    return lr * (0.1 ** int(epoch / 10))



if __name__ == "__main__":
    copy_data_from_gs()
    print(os.getcwd())
    print(os.listdir("."))

    X,y = load_data()
    model = trainer.models.cnn_model_2(IMG_SIZE)
    lr = 0.01
    rmsprop = keras.optimizers.rmsprop(lr=0.01,decay=0.01)
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.99, decay=0.001)
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
    
    os.system("gsutil -m cp model.h5 gs://pyeye_bucket/models/sunday_night.h5")
