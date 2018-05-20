# import the necessary packages
from imutils import face_utils
import os
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = {}
args["shape_predictor"] = "face_predictor.dat"
args["image"] = "tmp/101035.jpg"

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
imgs = os.listdir("../train_eye/images")



for img_path in imgs:
    image = cv2.imread(os.path.join("../train_eye/images/",img_path))
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array

        cp = image[rect.top():rect.bottom(),rect.left():rect.right()]
        cp = imutils.resize(cp, width=250, inter=cv2.INTER_CUBIC)
        cv2.imshow("rect", cp)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # image = image[rect[0]]
        # cv2.imshow("foobar", )
        # loop over the face parts individually
        tmp_imgs = []
        for (name, (i, j)) in list(filter(lambda x: x[0] in ["right_eye","left_eye"],face_utils.FACIAL_LANDMARKS_IDXS.items())):
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
            one_eye = {"roi" : roi, "clone":clone}
            tmp_imgs.append(one_eye)
            # show the particular face part
        for j,tmp_img in enumerate(tmp_imgs):
            cv2.imshow(str(j)+"1", tmp_img["roi"])
        cv2.waitKey(0)

        # visualize all facial landmarks with a transparent overlay
        output = face_utils.visualize_facial_landmarks(image, shape)
        cv2.imshow("Image", output)
        cv2.waitKey(0)