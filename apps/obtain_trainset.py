import sys, os
sys.path.insert(0, os.path.abspath('..'))
from webcam_utils.webcam import init_webcam, save_image
import mouse
from windows.cursor import queryMousePosition
from utils.constants import *



def on_click_event(cam):

    img_path = save_image(cam)
    posX, posY = queryMousePosition()
    with open(TRAIN_INDEX_PATH, 'a') as f:
        f.write("{0} {1} {2}\n".format(img_path, posX, posY))
	print("Got pic")



if __name__ == "__main__":
    try:
        cam = init_webcam()
        print("before")
        mouse.on_click(on_click_event,[cam])
        print("Registered smth")
        input()
    finally:
        mouse.unhook_all()

