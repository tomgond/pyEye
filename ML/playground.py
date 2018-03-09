import os
import pprint
from utils.constants import *
import numpy as np
import pandas as pd

import cv2




with open(os.path.join(PROJECT_HOME,"train_eye","indexes.txt"), "r") as infile:
    lines = infile.readlines()



lines = list(map(lambda x: list(map(lambda x: int(x.strip()), x.split(' ')[1:])), lines))
df = pd.DataFrame(lines)
df.astype(np.float64)
df.columns = ['x','y']
df.x = df.x.apply(lambda z: z+1920)


print(df.describe())






cv2.namedWindow('Dist', cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_FREERATIO)
blank_image = np.zeros((1080,3839,3), np.uint8)
for ind,row in df.iterrows():
    cv2.circle(blank_image, (row.x,row.y), 5, (0,255,0), -1)
cv2.imshow('Dist',blank_image)
cv2.imwrite("imbalance.jpg",blank_image)
cv2.waitKey(0)
