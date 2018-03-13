import os
import pprint
from ML.cv_classifiers import find_center
from skimage import transform
import ML.cv_classifiers as cv_classifier
from utils.constants import *
import numpy as np
import pandas as pd

import cv2




def find_center_that_i_look_at():
    with open(os.path.join(PROJECT_HOME,"train_eye","indexes.txt"), "r") as infile:
        lines = infile.readlines()



    lines = list(map(lambda x: list(map(lambda x: int(x.strip()), x.split(' ')[1:])), lines))
    df = pd.DataFrame(lines)
    df.astype(np.float64)
    df.columns = ['x','y']

    left_screen_middle = []
    right_screen_middle = []

    for _,coor in df.iterrows():
        if coor.x < 0:
            left_screen_middle.append((coor.x,coor.y))
        else:
            right_screen_middle.append((coor.x,coor.y))

    print(find_center(left_screen_middle)) # this is the center left
    print(find_center(right_screen_middle)) # this is the center right
    print(df.describe())
    cv2.namedWindow('Dist', cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_FREERATIO)
    blank_image = np.zeros((1080,3839,3), np.uint8)
    for ind,row in df.iterrows():
        cv2.circle(blank_image, (row.x,row.y), 5, (0,255,0), -1)
    cv2.imshow('Dist',blank_image)
    cv2.imwrite("imbalance.jpg",blank_image)
    cv2.waitKey(0)


def cropped_image_change():
    orig = cv2.imread("..\\train_eye\\images\\849.jpg")
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    img = cv_classifier.crop_face(img, 300)
    img = cv2.resize(img, (300, 300))
    cv2.imwrite("reduced.jpg", img)
    cv2.imwrite("orig.jpg", orig)

if __name__== "__main__":
    cropped_image_change()
