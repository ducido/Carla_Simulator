
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from test_carla import main



IM_WIDTH = 320
IM_HEIGHT = 240
def hough_transform(mask, image):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    masked_image = masked_image[:,:,:1].reshape(240,320)

    # Dilate the mask image
    kernel = np.ones((7, 7), np.uint8) 
    masked_image = cv2.dilate(masked_image, kernel, iterations=2)

    linesP = cv2.HoughLinesP(masked_image, 1, np.pi / 180, 100, None, 90, 60)

    blank = np.zeros_like(mask.reshape(240,320,1))
    print(blank.shape)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(blank, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)
    return blank

def process_img(image):
    image = image.reshape(1, IM_HEIGHT, IM_WIDTH, 3)

    pred = cv2.imread('data/train/masks/Town04_Clear_Noon_09_09_2020_14_57_22_frame_0.png', cv2.IMREAD_GRAYSCALE)
    pred = cv2.resize(pred, (320, 240)).reshape(240,320,1)
    blank = hough_transform(pred, image.reshape(240,320,3))
    plt.imshow(blank)
    plt.show()



image = cv2.imread('data/train/images/Town04_Clear_Noon_09_09_2020_14_57_22_frame_0.png')
image = cv2.resize(image, (320, 240))


process_img(image)




