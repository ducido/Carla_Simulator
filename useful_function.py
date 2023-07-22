import numpy as np
import cv2
from PIL import Image
import time

WIDTH = 128
HEIGHT = 128

def birdview_transform(img):
    """Apply bird-view transform to the image
    """
    a = 30
    b = 50
    src = np.float32([[0, HEIGHT//2], [WIDTH+0, HEIGHT//2], [0, HEIGHT], [WIDTH, HEIGHT]])
    dst = np.float32([[-a, 0], [WIDTH + a, 0], [b, HEIGHT], [WIDTH-b, HEIGHT]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    warped_img = cv2.warpPerspective(img, M, (WIDTH, HEIGHT)) # Image warping
    return warped_img

def hough_transform(image, lane_mask):
    # get lane
    lane = image * lane_mask
    lane = lane[:,:,:1].reshape(HEIGHT,WIDTH).astype('uint8') # input of hough

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

def find_left_right_points(image, points, draw=None):
    """Find left and right points of lane
    """

    # Consider the position 70% from the top of the image
    interested_line_y = int(HEIGHT * 0.7)
    if draw is not None:
        cv2.line(draw, (0, interested_line_y),
                 (WIDTH, interested_line_y), 255, 2)
    interested_line = image[interested_line_y, :]

    # cv2.imshow('', image)
    # cv2.waitKey(0)

    # Detect left/right points
    left_point = -1
    right_point = -1
    lane_width = 30
    center = WIDTH // 2

    # Traverse the two sides, find the first non-zero value pixels, and
    # consider them as the position of the left and right lines
    for x in range(center, 0, -1):
        # print(interested_line[x])
        if interested_line[x] > 0:
            left_point = x
            break
    for x in range(center + 1, WIDTH):
        if interested_line[x] > 0:
            right_point = x
            break

    if left_point == -1 and right_point == -1:
        left_point, right_point = points[-1]
    else: 
        # Predict right point when only see the left point
        if left_point != -1 and right_point == -1:
            right_point = left_point + lane_width

        # Predict left point when only see the right point
        if right_point != -1 and left_point == -1:
            left_point = right_point - lane_width
        
    points.append([left_point, right_point])


    # Draw two points on the image


    if draw is not None:
        if left_point != -1:
            cv2.circle(
                draw, (left_point, interested_line_y), 2, (0, 255 ,0), -1)
        if right_point != -1:
            cv2.circle(
                draw, (right_point, interested_line_y), 2, (0, 255 ,0), -1)

    return left_point, right_point, points

class PID:
    def __init__(self, kp= 1.0, ki= 0.0, kd= 0.0, setpoint= 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        # count time
        self.time_extract = time.monotonic

        # system status
        self._last_time = self.time_extract()
        self._last_state = None
        self._integral = 0

    def __call__(self, state):
        error = self.setpoint - state
        d_state = state - self._last_state if (self._last_state is not None) else state
        d_error = -d_state
        now = self.time_extract()
        d_time = now - self._last_time

        # check if d_time = 0
        d_time = d_time if (d_time) else 1e-12

        self._integral += self.ki * error * d_time
        proportional = self.kp * error
        derivative = (self.kd * d_error) / d_time

        # calculate control
        control_signal = proportional + self._integral + derivative

        # update state
        self._last_state = state
        self._last_time = now

        return control_signal
pid = PID(0.05, 0.0, 0.0, setpoint=0)
def calculate_control_signal(left_point, right_point):
    """Calculate speed and steering angle
    """


    # Calculate speed and steering angle
    # The speed is fixed to 50% of the max speed
    # You can try to calculate speed from turning angle
    throttle = 0.2
    steering_angle = 0
    im_center = WIDTH // 2

    if left_point != -1 and right_point != -1:

        # Calculate the deviation
        center_point = (right_point + left_point) // 2
        center_diff =  center_point - im_center

        # Calculate steering angle
        # You can apply some advanced control algorithm here
        # For examples, PID
        steering_angle = -pid(center_diff)

        # center_diff = -center_diff
        # steering_angle = -float(center_diff * 0.01)

    return throttle, steering_angle

def save(image, lane_mask, index):
    image = Image.fromarray(image.astype('uint8'))
    image.save(f'/{index}_image.png')
    lane_mask = Image.fromarray((lane_mask*255).reshape(HEIGHT,WIDTH).astype('uint8'))
    lane_mask.save(f'/{index}_lane_mask.png')

def cal_distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**.5

def cal_vecto(a,b):
    return [b[0] - a[0], b[1] - a[1]]

def cal_angle(target_list):

    A = target_list[-3]
    B = target_list[-2]
    C = target_list[-1]

    AB = cal_vecto(A, B)
    BC = cal_vecto(B, C)
    if (cal_distance(A, B) * cal_distance(B, C)) !=0:
        cos_alpha = (AB[0]*BC[0] + BC[1]*AB[1]) / (cal_distance(A, B) * cal_distance(B, C))
        alpha = np.abs(np.arccos(cos_alpha))
    
    else:
        alpha = 0
    # cos_alpha = (AB[0]*BC[0] + BC[1]*AB[1]) / (cal_distance(A, B) * cal_distance(B, C))

    if C[0] - B[0] < 0:
        alpha = -alpha
    return 0.5* (alpha/np.pi)




def find_target_points(image, points, draw=None):
    """Find left and right points of lane
    """

    # Consider the position 70% from the top of the image
    interested_line_y = int(HEIGHT * 0.7)
    if draw is not None:
        cv2.line(draw, (0, interested_line_y),
                 (WIDTH, interested_line_y), 255, 2)
    interested_line = image[interested_line_y, :]

    # Detect left/right points
    left_point = [-1, interested_line_y]
    right_point = [-1, interested_line_y]
    lane_width = 30
    center = WIDTH // 2

    # Traverse the two sides, find the first non-zero value pixels, and
    # consider them as the position of the left and right lines
    for x in range(center, 0, -1):
        # print(interested_line[x])
        if interested_line[x] > 0:
            left_point[0] = x
            break
    for x in range(center + 1, WIDTH):
        if interested_line[x] > 0:
            right_point[0] = x
            break

    if (left_point[0] == -1 and right_point[0] == -1):
        left_point, right_point = points[-1]
    else: 
        # Predict right point when only see the left point
        if left_point[0] != -1 and right_point[0] == -1:
            right_point[0] = left_point[0] + lane_width

        # Predict left point when only see the right point
        if right_point[0] != -1 and left_point[0] == -1:
            left_point[0] = right_point[0] - lane_width

    # if len(points) != 0:
    #     if (np.abs(left_point[0] - points[-1][0][0]) > 5 or np.abs(right_point[0] - points[-1][1][0]) > 5):
    #         left_point, right_point = points[-1]
        
    points.append([left_point, right_point])

    # Draw two points on the image


    if draw is not None:
        if left_point != -1:
            cv2.circle(
                draw, (left_point[0], interested_line_y), 2, (0, 255 ,0), -1)
        if right_point != -1:
            cv2.circle(
                draw, (right_point[0], interested_line_y), 2, (0, 255 ,0), -1)

    return left_point, right_point, points