import os
import cv2
import trainer.cnn as cnn

if __name__ == "__main__":
    imgs_files = os.listdir("images")
    try:
        os.mkdir("output_VGG")
    except:
        pass
    for img_file in imgs_files:
        try:
            if not(os.path.exists(("images/{0}".format(img_file))) and not os.path.exists(("images/{0}".format(img_file)))):
                img = cv2.imread("images/{0}".format(img_file))
                img = cnn.img_reprocess(img, crop_size=250, img_size=224, to_greyscale=False)
                cv2.imwrite("output_VGG/{0}".format(img_file),img)
            else:
                print("Passed")
        except:
            print("Error")

