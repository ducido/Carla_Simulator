
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from test_carla import lane_detect
from useful_function import *

width = 128
height = 128


i = np.random.choice(range(0, 4929))
print(i)
# i = 4888
image = cv2.imread(f'data/val/images/{i}.png')
image = cv2.resize(image, (width, height))


mask = lane_detect(image.reshape(1, height, width, 3))
mask = mask[:,:,:,1:2].reshape(height, width, 1)
mask[mask >= 0.6] = 1
mask[mask < 0.6] = 0

# mask = cv2.imread(f'data/val/masks/{i}.png')
# mask = cv2.resize(mask, (width, height))[:,:,:1]


image_bird = birdview_transform(mask)

draw = image.copy()
draw = birdview_transform(draw)
left, right  = find_left_right_points(image_bird, draw= draw)
print(left, right)

plt.subplot(2,3,1)
plt.imshow(image)

plt.subplot(2,3,2)
plt.imshow(mask)

plt.subplot(2,3,3)
plt.imshow(draw)

plt.subplot(2,3,4)
plt.imshow(image_bird * 255)

plt.show()







