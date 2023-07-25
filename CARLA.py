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
import cv2
import tensorflow as tf
from PIL import Image
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from useful_function import *
import argparse
import json
from src.frontend import Segment

argparser = argparse.ArgumentParser(
    description='Evaluate Road Segmentation Model')

argparser.add_argument(
    '-c',
    '--conf', default="config_UNet.json",
    help='path to configuration file')


args = argparser.parse_args()
config_path = args.conf

# Open and load the config json
with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())

############################################### call pretrained model
backend = config["model"]["backend"]
input_size = (config["model"]["im_width"], config["model"]["im_height"])
classes = config["model"]["classes"]

# define the model and train
segment = Segment(backend, input_size, classes)
model = segment.feature_extractor

# Load best model
model.load_weights(config['test']['model_file'])

############################################### main logic
class Carla:
    WIDTH = 128
    HEIGHT = 128
    CAM_WIDTH = 320
    CAM_HEIGHT = 240
    index = 0
    SHOW_CAM = True
    SAVE = False
    VIEW = False
    points = []
    target_list = []

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.load_world('Town04')
        blueprint_library = self.world.get_blueprint_library()
        self.model_3 = blueprint_library.filter('model3')[0]

    def reset(self):
        self.actor_list = []

        ###### get random position of car
        self.transform = np.random.choice(self.world.get_map().get_spawn_points())
        # self.transform = carla.Transform(carla.Location(x=-7.530000, y=208.919998, z=0.500000), carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        print(self.transform)

        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        # self.vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)

        ##### define camera RGB
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.WIDTH}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.HEIGHT}')
        self.rgb_cam.set_attribute('fov', '110')

        # attach camera RGB to car
        transform = carla.Transform(carla.Location(x=2.1, z=1.6))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)

        ##### define camera RGB behind car
        self.rgb_cam_behind = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam_behind.set_attribute('image_size_x', f'{self.CAM_WIDTH}')
        self.rgb_cam_behind.set_attribute('image_size_y', f'{self.CAM_HEIGHT}')
        self.rgb_cam_behind.set_attribute('fov', '110')

        # attach camera RGB to car
        transform_behind = carla.Transform(carla.Location(x=-6, z=4))
        self.sensor_behind = self.world.spawn_actor(self.rgb_cam_behind, transform_behind, attach_to=self.vehicle)

        self.sensor.listen(lambda image: self.process_img(image))
        self.actor_list.append(self.sensor)
        self.sensor_behind.listen(lambda image: self.view_car(image))
        self.actor_list.append(self.sensor)

        time.sleep(120)
        self.destroy()

    def destroy(self):
        for actor in self.actor_list:
            actor.destroy()
        print('All cleaned up!')

    def view_car(self, image):
        if self.VIEW:
            image = np.array(image.raw_data)
            image = image.reshape(1, self.CAM_HEIGHT, self.CAM_WIDTH, 4)[:,:,:,:3].reshape(self.CAM_HEIGHT, self.CAM_WIDTH, 3)
            cv2.imshow('car', image)
            cv2.waitKey(1)

    def process_img(self, image):
        # transform data to detect
        image = np.array(image.raw_data)
        image = image.reshape(1, self.HEIGHT, self.WIDTH, 4)[:,:,:,:3]

        # detect
        lane_mask = model.predict(image)
        lane_mask = lane_mask[:,:,:,1:2].reshape(self.HEIGHT, self.WIDTH, 1)
        lane_mask[lane_mask >= 0.7] = 1
        lane_mask[lane_mask < 0.7] = 0

        lane_mask_top = birdview_transform(lane_mask)

        image = image.reshape(self.HEIGHT, self.WIDTH, 3)
        draw = image.copy()
        draw = birdview_transform(draw)

        left, right, self.points = find_left_right_points(lane_mask_top, self.points, draw)
        throttle, steering_angle = calculate_control_signal(left, right)
        print(f"throttle: {throttle}, steer: {steering_angle}")

        # self.vehicle.apply_control(carla.VehicleControl(throttle, steering_angle))
        self.vehicle.set_autopilot(True)

        if self.SHOW_CAM:
            cv2.imshow("image", image)
            cv2.imshow("lane_mask", lane_mask)
            # cv2.imshow("lane_mask", lane_hough)
            # cv2.imshow("lane_mask_top", lane_mask_top)
            # cv2.imshow("draw", draw)
            cv2.waitKey(1)
            pass
        
        if self.SAVE: 
            save(image, lane_mask, self.index)
            self.index += 1
    
    def run(self):
        self.reset()

carla_world = Carla()
carla_world.run()
