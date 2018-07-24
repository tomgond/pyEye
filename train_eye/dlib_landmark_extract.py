# import the necessary packages
from imutils import face_utils
import os
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
args = {}


args["shape_predictor"] = "../trainer/face_predictor.dat"
# args["shape_predictor"] = r'D:\Programs\OpenFace\OpenFace_2.0.0_win_x64\model\patch_experts\cen_patches_1.00_of.dat'

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])




def dlib_preprocess(image):
    image = imutils.resize(image,width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray,1)

    # loop over the face detections
    i = 0
    print(len(rects))
    rect = rects[0]
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array

    cp = image[rect.top():rect.bottom(), rect.left():rect.right()]
    cp = cv2.resize(cp, (250,250))
    # cv2.imshow("rect", cp)

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    # image = image[rect[0]]
    # cv2.imshow("foobar", )
    # loop over the face parts individually
    tmp_imgs = {}
    for (name, (i, j)) in list(
            filter(lambda x: x[0] in ["right_eye", "left_eye"], face_utils.FACIAL_LANDMARKS_IDXS.items())):
        # clone the original image so we can draw on it, then
        # display the name of the face part on the image
        clone = image.copy()
        # loop over the subset of facial landmarks, drawing the
        # specific face part
        # for (x, y) in shape[i:j]:
        #     cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
        # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = image[y:y + h, x:x + w]
        roi = cv2.resize(roi, (150, 100))
        one_eye = {"roi": roi, "clone": clone}
        tmp_imgs[name] = one_eye
        # show the particular face part

    return {"right_eye_input":tmp_imgs['right_eye']['roi'], "left_eye_input":tmp_imgs['left_eye']['roi'], "face_input":cp}
    # os.mkdir(os.path.join('output_landmarks', img_path))
    # for im_name, cropped_imgs in tmp_imgs.items():
    #     cv2.imwrite(os.path.join('output_landmarks', img_path, im_name + ".jpg"), cropped_imgs['roi'])
    # cv2.imwrite(os.path.join('output_landmarks', img_path, 'face.jpg'), cp)
    # print("Finished : {0}".format(img_path))


if __name__=="__main__":
    imgs = os.listdir("images")
    for outer_index, img_path in enumerate(imgs):
        if outer_index % 100 == 0:
            print("At index : {0}".format(outer_index))
        if os.path.exists(os.path.join('output_landmarks', img_path)):
            print("Skipping image {0} because it already exists".format(img_path))
            continue
        try:

            image = cv2.imread(os.path.join("images/",img_path))
            ret_dlib = dlib_preprocess(image)
            for im_name, cropped_imgs in ret_dlib.items():
                cv2.imwrite(os.path.join('output_landmarks', img_path, im_name + ".jpg"), cropped_imgs[im_name])
        except:
            print("Could not find face for image: {0}".format(img_path))