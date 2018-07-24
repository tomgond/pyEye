import os
import numpy as np
from train_eye import dlib_landmark_extract
from keras import Model
from keras.applications import VGG16
from trainer.models import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import load_model
from ML.cv_classifiers import find_center
from ML.cnn import cnn_model, prepare_image
from keras.preprocessing.image import img_to_array, ImageDataGenerator
import cv2
from webcam_utils.webcam import init_webcam
from trainer import cnn
import trainer
import sys
sys.path.append('C:\\Programs\\Anaconda2\\envs\\tensorflow_3.5_new\Lib\\site-packages')
import win32api

def click(x,y):
    win32api.SetCursorPos((x,y))

def predict_one_image(x):
    x,y = mdl.predict(x)[0]
    return x,y


def predict_mul_images(x):
    predictions = []
    for _ in range(0, 10):
        aug_img = train_datagen.flow(x).next()
        aug_img.reshape((1,1,300,300))
        prediction = mdl.predict(aug_img)[0]
        predictions.append(prediction)
    x,y = find_center(predictions)
    return x,y

webcam = init_webcam()

train_datagen = ImageDataGenerator(
    width_shift_range=0.3,
    height_shift_range=0.3,
    data_format="channels_first",
    horizontal_flip=False,
    fill_mode='nearest')



params = {}
params['conv_funcs'] = 40
params['kernel_size'] = (2, 2)
params['max_pooling'] = (4, 4)
params['optimizer'] = 'adadelta'
params['dropout'] = 0.1
params['initial_final_layer'] = 3000
params['final_layers'] = 3
params['final_layer_decrease_factor'] = 0.7
params['lr'] = 0.5

model = trainer.models.cnn_model_multiple_inputs(params=params)
# model.load_weights("baised_to_left_screen.h5")
model.load_weights("shit_too_much_tries.h5")




while True:
    try:
        input("say_smth")
        print("saied")
        avgs = []
        for i in range(0,5):
            try:
                ret, img = webcam.read()
                ret, img = webcam.read()
                cv2.imshow("image", img)
                cv2.waitKey(1)
                img = dlib_landmark_extract.dlib_preprocess(img)
                # cv2.namedWindow('foobar')
                # cv2.imshow('foobar', img)
                # cv2.waitKey(0)

                # x = x.reshape((1,) + x.shape)
                for key in img:
                    x = np.rollaxis(img[key].reshape((1,) + img[key].shape), 3, 1)
                    # zz = train_datagen.flow(x).next()
                    img[key] = x
                avgs.append(model.predict(img)[0])
            except Exception as e:
                print(e)
        x,y = find_center(avgs)
        print("{0} {1}".format(int(x),int(y)))

        click(int(x), int(y))
    except Exception as e:
        print(e)
        # pass

print(mdl)
