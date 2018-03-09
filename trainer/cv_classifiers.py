import numpy as np
import pprint
from pprint import pprint
import statistics
import cv2
import os
import random




upper_body= cv2.CascadeClassifier('haarcascade_mcs_upperbody.xml')
frontal2 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
frontaldefault = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
left_eye = cv2.CascadeClassifier('haarcascade_mcs_lefteye.xml')

classifiers = [upper_body, frontal2, frontaldefault, mouth, left_eye]
# cas_temp = {}
# cas_temp["haarcascade_mcs_lefteye.xml"] = cascades["haarcascade_mcs_lefteye.xml"]
# cas_temp['haarcascade_mcs_mouth.xml'] = cascades['haarcascade_mcs_mouth.xml']

# cascades = cas_temp
# to test - lefteye
# to test - mouth

def load_all_cascades_in_dir():
    cascades = {}
    files = os.listdir(".")
    for cas in filter(lambda x: ".xml" in x, files):
        try:
            cas_model = cv2.CascadeClassifier(cas)
            cascades[cas] = cas_model
        except Exception as e:
            "Cannot create model from {0}".format(cas)
    return cascades


def has_upper_body(img):
    cas_res = upper_body.detectMultiScale(img, 1.3, 5)
    front_res = frontal2.detectMultiScale(img, 1.3, 5)
    front_default_res = frontaldefault.detectMultiScale(img, 1.3, 5)
    mouth_res = mouth.detectMultiScale(img, 1.3, 5)
    l_eye_res = left_eye.detectMultiScale(img, 1.3, 5)
    if (len(l_eye_res)+ len(mouth_res) +len(cas_res) + len(front_res) + len(front_default_res)) > 0 :
        return True

def test_specific_haar_classifiers():
    total_images = 0
    recognized_in = 0
    for img_path in os.listdir("../train_eye/images"):
        total_images += 1
        if total_images % 50 == 0:
            print recognized_in
            print total_images
            print "Total recognized : {0}".format(float(recognized_in) / total_images)
        img_path = os.path.join("../train_eye/images",img_path)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if has_upper_body(img):
            recognized_in +=1
    print "Total recognized : {0}".format(float(recognized_in)/total_images)
    print "Finished"


def find_center(coor_list):
    # x_sum = 0
    # y_sum = 0
    # total = 0
    x_s = []
    y_s = []
    for coor in coor_list:
        # total += 1
        x_s.append(coor[0])
        y_s.append(coor[1])
        # x_sum += coor[0]
        # y_sum += coor[1]
    return (statistics.median(x_s), statistics.median(y_s))



def crop_face(img, crop_size):
    centers = []
    for clas in classifiers:
        detect = clas.detectMultiScale(img, 1.3, 5)
        for x,y,w,h in detect:
            # cv2.rectangle(img, (x + (w / 2), y + (h / 2)), (x + (w / 2) + 10, y + (h / 2) + 10), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
            centers.append((x+(w/2), y+(h/2)))
    center = find_center(centers)
    topleft_X = max(int(center[0] - crop_size/2),0)
    topleft_Y = max(int(center[1] - crop_size/2),0)
    if (topleft_X + crop_size) > img.shape[1]:
        topleft_X = img.shape[1] - crop_size
    if (topleft_Y + crop_size) > img.shape[0]:
        topleft_Y = img.shape[0] - crop_size

    # cv2.rectangle(img, (topleft_X, topleft_Y), (topleft_X+crop_size, topleft_Y+crop_size), (33,33,33),2)

    cropped_image = img[topleft_Y:topleft_Y+crop_size,topleft_X:topleft_X+crop_size]
    return cropped_image

if __name__ == "__main__":


    for i,img_path in enumerate(os.listdir("../train_eye/images")):
        img_path = os.path.join("../train_eye/images",img_path)
        img = cv2.imread(img_path)
        print i
        crop_face(img, 300)






    '''
    test_specific_haar_classifiers()


    total_images = 0
    reses = {}
    for i,img_path in enumerate(os.listdir("../train_eye/images")):
        if i%50 == 0:
            print reses
        total_images += 1
        img_path = os.path.join("../train_eye/images",img_path)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print "======================="
        for cas_name, cas_m in cascades.items():
            cas_recognized = cas_m.detectMultiScale(img, 1.3, 5)
            if len(cas_recognized):
                # print cas_name
                reses[cas_name] = reses.get(cas_name,0) + 1
            for (x, y, w, h) in cas_recognized:
                cv2.rectangle(img, (x, y), (x + w, y + h), (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # raw_input("foobar")
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     roi_gray = img[y:y + h, x:x + w]
    #     roi_color = img[y:y + h, x:x + w]
    #     eyes = eye_cascade.detectMultiScale(roi_gray)
    #     for (ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    '''
