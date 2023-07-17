import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
from test_carla import lane_detect
import cv2
import tensorflow as tf
from PIL import Image

# index = 0

# def save(pred, blank, index):
#     pred = Image.fromarray((pred*255).reshape(240,320).astype('uint8'))
#     pred.save(f'save_image/{index}_pred.png')
#     blank = Image.fromarray((blank*255).reshape(240,320).astype('uint8'))
#     blank.save(f'save_image/{index}_blank.png')
#     index +=1

# def hough_transform(mask, image):
#     # masked_image = cv2.bitwise_and(image, image, mask=mask)
#     masked_image = mask * image
#     masked_image = masked_image[:,:,:1].reshape(240,320).astype('uint8')

#     # Dilate the mask image
#     # kernel = np.ones((5, 5), np.uint8) 
#     # masked_image = cv2.dilate(masked_image, kernel, iterations=1)
#     # cv2.imshow('mask', masked_image)
#     # cv2.waitKey(1)
#     linesP = cv2.HoughLinesP(masked_image, 1, np.pi / 180, 100, None,minLineLength=10, maxLineGap=20)

#     blank = np.zeros_like(mask.reshape(240, 320, 1))
#     if linesP is not None:
#         for i in range(0, len(linesP)):
#             l = linesP[i][0]
#             cv2.line(blank, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)
#     return blank

# def process_img(image):
#     global index
#     image = np.array(image.raw_data)
#     image = image.reshape(1, 240, 320, 4)
#     image = image[:, :, :, :3]
#     # cv2.imshow('i', image.reshape(240, 320, 3))
#     # cv2.waitKey(1)
    
#     pred = lane_detect(image)
#     pred = pred[:,:,:,1:2].reshape(240, 320, 1)

#     cv2.imshow('pred', pred)
#     cv2.waitKey(1)

#     blank = hough_transform(pred, image.reshape(240,320,3))
#     # cv2.imshow('', blank)
#     # cv2.waitKey(1)

#     save(pred, blank, index)
#     pass


class Carla:
    WIDTH = 128
    HEIGHT = 128
    index = 0
    SHOW_CAM = True
    SAVE = False

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)
        
        self.world = self.client.load_world('Town04')

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = self.world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        #print(blueprint_library.filter('vehicle'))
        self.model_3 = blueprint_library.filter('model3')[0]

    def reset(self):
        self.actor_list = []

        # get random position of car
        self.transform = np.random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)

        # define camera RGB
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.WIDTH}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.HEIGHT}')
        self.rgb_cam.set_attribute('fov', '110')

        # attach camera RGB to car
        transform = carla.Transform(carla.Location(x=2.3, z=1.1))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)

        # show camera and detect lane
        self.sensor.listen(lambda image: self.process_img(image))
        self.actor_list.append(self.sensor)

        time.sleep(90)
        self.destroy()

    def destroy(self):
        for actor in self.actor_list:
            actor.destroy()
        print('All cleaned up!')

    def process_img(self, image):
        # transform data to detect
        image = np.array(image.raw_data)
        image = image.reshape(1, self.HEIGHT, self.WIDTH, 4)[:,:,:,:3]

        # detect
        lane_mask = lane_detect(image)
        lane_mask = lane_mask[:,:,:,1:2].reshape(self.HEIGHT, self.WIDTH, 1)
        # lane_mask = np.concatenate((lane_mask, lane_mask, lane_mask), axis= 2)

        # # hough transform
        image = image.reshape(self.HEIGHT, self.WIDTH, 3)
        # lane_dilate, lane_hough = self.hough_transform(image, lane_mask)

        # show cam
        if self.SHOW_CAM:
            cv2.imshow('image', image)
            cv2.imshow("lane_mask", lane_mask)
            cv2.waitKey(2)
        if self.SAVE:
            self.save(image, lane_mask)

    def hough_transform(self, image, lane_mask):
        # get lane
        lane = image * lane_mask
        lane = lane[:,:,:1].reshape(self.HEIGHT,self.WIDTH).astype('uint8') # input of hough

        # dilate lane
        kernel = np.ones((3, 3), np.uint8) 
        lane_dilate = cv2.dilate(lane, kernel, iterations=1)

        # hough transform
        linesP = cv2.HoughLinesP(lane_dilate, 1, np.pi / 180, 100, None,minLineLength=10, maxLineGap=30)

        lane_hough = np.zeros_like(lane_mask)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(lane_hough, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)
        return lane_dilate, lane_hough
    
    def save(self,image, lane_mask):

        image = Image.fromarray(image.astype('uint8'))
        image.save(f'save_image/{self.index}_image.png')
        lane_mask = Image.fromarray((lane_mask*255).reshape(self.HEIGHT,self.WIDTH).astype('uint8'))
        lane_mask.save(f'save_image/{self.index}_lane_mask.png')
        self.index +=1
    
    def run(self):
        self.reset()

carla_world = Carla()
carla_world.run()

# try:
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(10.0)

#     # world = client.get_world()
#     world = client.load_world('Town04')
    
#     blueprint_library = world.get_blueprint_library()

#     bp = blueprint_library.filter("model3")[0]
#     # print(world.get_map().get_spawn_points())
#     spawn_point = np.random.choice(world.get_map().get_spawn_points())
#     vehicle = world.spawn_actor(bp, spawn_point)
#     vehicle.set_autopilot(True)
#     actor_list.append(vehicle)

#     cam_bp = blueprint_library.find('sensor.camera.rgb')
#     cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
#     cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
#     cam_bp.set_attribute('fov', '110')

#     # Adjust sensor relative to vehicle
#     spawn_point = carla.Transform(carla.Location(x = 2.5,z=2))

#     # spawn the sensor and attach to vehicle.
#     sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
#     sensor.listen(lambda data: process_img(data))

#     actor_list.append(sensor)
#     time.sleep(120)

# finally:
#     for actor in actor_list:
#         actor.destroy()
#     print('All cleaned up!')
    