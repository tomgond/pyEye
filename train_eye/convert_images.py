import os
import cv2
import sys
sys.path.insert(0, os.path.abspath('..'))
import trainer.cnn as cnn

if __name__ == "__main__":
    imgs_files = os.listdir("images")
    try:
        os.mkdir("output_VGG")
    except:
        pass
    i = 0
    for img_file in imgs_files:
        try:
            if not(os.path.exists(("images/{0}".format(img_file))) and not os.path.exists(("images/{0}".format(img_file)))):
                img = cv2.imread("images/{0}".format(img_file))
                img = cnn.img_reprocess(img, crop_size=224, img_size=224, to_greyscale=False)
                cv2.imwrite("output_VGG/{0}".format(img_file),img)
                i+=1
                if i%1000 == 0:
                    print("Converting... i = {0}".format(i))
            else:
                print("Passed")
        except:
            print("Error")

