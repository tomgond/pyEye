import os
import numpy as np
from sklearn.preprocessing import StandardScaler

from openface_interaction.openface_learning import get_dataset
from train_eye import dlib_landmark_extract
from keras import Model
from keras.applications import VGG16
from trainer.models import *
from sklearn.externals import joblib
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
import subprocess

OPENFACE_DIR = r"D:\Programs\OpenFace\OpenFace_2.0.0_win_x64"
OPENFACE_FEATURE_EXTRACTOR_PROC = "FeatureExtraction.exe"
open_face_bin = os.path.join(OPENFACE_DIR, OPENFACE_FEATURE_EXTRACTOR_PROC)
INDEXES_FILE = r"D:\Programming\pyEye\train_eye\indexes.txt"


def click(x,y):
    win32api.SetCursorPos((x,y))


webcam = init_webcam()


def get_random_forest_classifier_x():
    return joblib.load("models/clf_randomforest_x.pkl")

def get_random_forest_classifier_y():
    return joblib.load("models/clf_randomforest_y.pkl")

def get_pca():
    return joblib.load("models/pca.pkl")

def get_scaler():
    return joblib.load("models/scalar.pkl")

def get_label_encoder():
    return joblib.load("models/label_encoder.pkl")

def get_random_forest_classifier():
    return joblib.load("models/clf_randomforestclassifier.pkl")

def extract_features(infile, outdir):
    FNULL = open(os.devnull, 'w')
    subprocess.call([open_face_bin, "-f", infile, "-out_dir",
                         outdir], stdout=FNULL)


pca = get_pca()
clf = get_random_forest_classifier()
label_encoder = get_label_encoder()
clf_x = get_random_forest_classifier_x()
clf_y = get_random_forest_classifier_y()
scaler = get_scaler()

while True:
    try:
        input("say_smth")
        print("saied")
        avgs = []
        ret, img = webcam.read()
        ret, img = webcam.read()
        cv2.imwrite("tmp_image.jpg", img)
        extract_features("t"
                         ""
                         "mp_image.jpg", r"D:\Programming\pyEye\openface_interaction")
        X, _, _ = get_dataset("tmp_image.csv")
        X = scaler.transform(X)
        X = pca.transform(X)
        lbl = clf.predict(X)
        coors = label_encoder.inverse_transform(lbl)
        print(coors)
        click(int(coors[0][0]+480), int(coors[0][1]+270))


    except Exception as e:
        print(e)


'''


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
'''