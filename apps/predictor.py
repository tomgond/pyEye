import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import load_model
from ML.cv_classifiers import find_center
from ML.cnn import cnn_model, prepare_image
from keras.preprocessing.image import img_to_array, ImageDataGenerator
import cv2
from webcam_utils.webcam import init_webcam
import win32api, win32con


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
mdl = cnn_model()
mdl.load_weights("../ML/model_702.h5")

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    data_format="channels_first",
    horizontal_flip=False,
    fill_mode='nearest')



while True:
    input()
    ret, img = webcam.read()
    ret, img = webcam.read()
    img = prepare_image(img)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x,y = predict_one_image(x)
    print("{0} {1}".format(int(x),int(y)))
    click(int(x), int(y))

print(mdl)
