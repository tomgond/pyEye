import os

from keras import Model
from keras.applications import VGG16

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import load_model
from ML.cv_classifiers import find_center
from ML.cnn import cnn_model, prepare_image
from keras.preprocessing.image import img_to_array, ImageDataGenerator
import cv2
from webcam_utils.webcam import init_webcam
from trainer import cnn
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
mdl = cnn.get_my_prediction_model()
mdl.load_weights("my_model.h5")

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    data_format="channels_first",
    horizontal_flip=False,
    fill_mode='nearest')

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(3, 224, 224))
base_model.summary()

VGG_convolution_only = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
VGG_convolution_only._make_predict_function()

pred_model = cnn.get_my_prediction_model()

while True:
    input()
    ret, img = webcam.read()
    ret, img = webcam.read()
    try:
        img = cnn.img_reprocess(img, crop_size=224, img_size=224, to_greyscale=False)
    except:
        print("Can't find face! next...")
        continue
    # cv2.namedWindow('foobar')
    # cv2.imshow('foobar', img)
    # cv2.waitKey(0)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x,y = mdl.predict(VGG_convolution_only.predict(x))[0]
    print("{0} {1}".format(int(x),int(y)))
    click(int(x), int(y))

print(mdl)
