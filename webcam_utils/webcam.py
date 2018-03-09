import cv2
import random
import os
from utils.constants import *

def init_webcam():
    return cv2.VideoCapture(0)


def save_image(cam):
    '''
    
    :param cam: 
    :return: path to saved image 
    '''
    ret, frm = cam.read()
    img_name =  "{0}.jpg".format(int(random.random()*100000000))
    res = cv2.imwrite(os.path.join(TRAIN_IMAGES_PATH, img_name), frm)
    return img_name





'''

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

'''