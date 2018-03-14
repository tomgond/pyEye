import os
import cv2
import trainer.cnn as cnn

if __name__ == "__main__":
    imgs_files = os.listdir("images")
    try:
        os.mkdir("output")
    except:
        pass
    for img_file in imgs_files:
        try:
            img = cv2.imread("images/{0}".format(img_file))
            img = cnn.img_reprocess(img, crop_size=250, img_size=150)
            cv2.imwrite("output/{0}".format(img_file),img)
        except:
            pass

