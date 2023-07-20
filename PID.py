import queue
import numpy as np
import cv2
import queue
import carla
import os
import math
import sys
import glob
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


def get_speed(vehicle):

    vel = vehicle.get_velocity()
    return 3.6*math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

class VehiclePIDController:

    def __init__(self,vehicle, args_lateral, args_longtitudnal, max_throttle = 0.75, max_break = 0.3, max_steering = 0.8 ):
        self.max_break = max_break
        self.max_steering = max_steering
        self.max_throttle = max_throttle

        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.past_steering = self.vehicle.get_control().steer
        self.long_controller = PIDLongtitudmalControl(self.vehicle, **args_longtitudnal)
        self.lat_controller = PIDLateralControl(self.vehicle, **args_lateral)
    
    def run_step(self, target_speed, waypoint):
        acceleration = self.long_controller.run_step(target_speed)
        current_steering = self.lat_controller.run_step(waypoint)
        control = carla.VehicleControl()
        if acceleration == 0.8:
            control.throttle = min(abs(acceleration), self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_break)

        if current_steering > self.past_steering+0.1:
            current_steering = self.past_steering+0.1
        elif current_steering < self.past_steering-0.1:
            current_steering = self.past_steering - 0.1
        
        if current_steering >=0:
            steering = min(self.max_steering, current_steering)
        else:
            steering = max(-self.max_steering, current_steering)

        control_steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering


        return control
    



class PIDLongtitudmalControl():

    def __init__(self, vehicle, k_p = 1.0, k_i = 0.0, k_d =0.0, dt = 0.03 ):
        self.vehicle = vehicle
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.error_buffer = queue.deque(maxlen = 10)
        self.dt = dt 

    def run_step(self, target_speed):
        current_speed = get_speed(self.vehicle)
        return self.pid_controller(target_speed, current_speed)
    
    def pid_controller(self, target_speed, current_speed):
        error = target_speed - current_speed
        self.error_buffer.append(error)
        if (len(self.error_buffer)>=2):
            de = (self.error_buffer[-1] - self.error_buffer[-2])/self.dt 
            ie = sum(self.error_buffer)/self.dt
        else:
            de = 0
            ie = 0
        
        return np.clip(self.k_p*error + self.k_d*de + self.k_i*ie, -1.0,1.0)



class PIDLateralControl():
    def __init__(self, vehicle, k_p = 1.0, k_i = 0.0, k_d =0.0, dt = 0.03 ):
        self.vehicle = vehicle
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.dt = dt
        self.error_buffer = queue.deque(maxlen = 10)

    def run_step(self, waypoint):
        return self.pid_controller(waypoint, self.vehicle.get_transform())
    
    def pid_controller(self, waypoint, vehicle_transform):
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x = math.cos(math.radians(vehicle_transform.rotation.yaw)), y = math.sin(math.radians(vehicle_transform.rotation.yaw)))
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x - v_begin.x, waypoint.transform.location.y - v_begin.y, 0.0])
        dot = math.acos(np.clip(np.dot(w_vec, v_vec)/np.linalg.norm(w_vec)*np.linalg.norm(v_vec), -1.0, 1.0))
        cross = np.cross(v_vec, w_vec)
        if cross[2] == 0:
            dot *= -1
        self.error_buffer.append(dot)

        if len(self.error_buffer) >=2:
            de = (self.error_buffer[-1] - self.error_buffer[-2]) / self.dt
            ie = sum(self.error_buffer)*self.dt

        else:
            de = 0.0
            ie = 0.0
        
        return np.clip(self.k_p*dot + self.k_d*de + self.k_i*ie, -1.0,1.0)
    
HEIGHT = 200
WIDTH = 200
def process_img(image):
    image = np.array(image.raw_data)
    image = image.reshape(1, HEIGHT, WIDTH, 4)[:,:,:,:3]
    cv2.imshow('', image.reshape(HEIGHT, WIDTH, 3))
    cv2.waitKey(0)

def main():
    actor_list = []
    try :
        client = carla.Client('localhost',2000)
        client.set_timeout(5.0)
        world = client.get_world()
        map = world.get_map()

        blueprint_library = world.get_blueprint_library()

        vehicle_bp = blueprint_library.filter('model3')[0]

        spawnpoint = np.random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp , spawnpoint)

        actor_list.append(vehicle)
        control_vehicle = VehiclePIDController(vehicle, args_lateral = {'k_p':1, 'k_d':0, 'k_i':0}, args_longtitudnal ={'k_p':1, 'k_d':0, 'k_i':0} )
        while True:
            waypoints = world.get_map().get_waypoint(vehicle.get_location())
            waypoints = np.random.choice(waypoints.next(0.3))
            control_signal = control_vehicle.run_step(5, waypoints)

            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '200')
            camera_bp.set_attribute('image_size_y', '200')
            camera_transform = carla.Transform(carla.Location(x=1.5, y = 2.4))
            camera_bp.set_attribute('fov', '90')
            camera = world.spawn_actor(camera_bp, camera_transform)
            camera.listen(lambda image: process_img(image))

            actor_list.append(camera)


            # depth_camera_bp = blueprint_library.find('sensor.camera.depth')
            # depth_camera_transform = carla.Transform(carla.Location(x=1.5, y = 2.4))
            # depth_camera = world.spawn_actor(depth_camera_bp, depth_camera_transform)
            # depth_camera.listen()

            # actor_list.append(depth_camera)
            time.sleep(10)

    finally:
        for actor in actor_list:
            actor.destroy()
        print('All cleaned up!')


if __name__ == '__main__':
    main()
        
        