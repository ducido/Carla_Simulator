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

def birdview_transform2(img):
    """Apply bird-view transform to the image
    """
    a = 30
    b = 50
    src = np.float32([[50, HEIGHT * 0.6], [78, HEIGHT * 0.6], [50, HEIGHT * 0.8], [78, HEIGHT*0.8]])
    dst = np.float32([[0, 0], [0, WIDTH], [0, HEIGHT], [WIDTH, HEIGHT]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    warped_img = cv2.warpPerspective(img, M, (WIDTH, HEIGHT)) # Image warping
    return warped_img

def hough_transform(image, lane_mask):
    # get lane
    lane = image * lane_mask
    lane = lane[:,:,:1].reshape(HEIGHT,WIDTH).astype('uint8') # input of hough
    # cv2.imshow('', lane)
    # cv2.waitKey(1)
    # dilate lane
    # kernel = np.ones((3, 3), np.uint8) 
    # lane_dilate = cv2.dilate(lane, kernel, iterations=1)

    # hough transform

    linesP = cv2.HoughLinesP(lane, 1, np.pi / 180, 10, None, 10, 20)
    lane_hough = np.zeros_like(lane)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(lane_hough, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)
    return lane_hough

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
    lane_width = 29
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

    # Predict right point when only see the left point
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width

    # Predict left point when only see the right point
    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width
        
    # if len(points) != 0:
    #     # if (np.abs(left_point - points[-1][0]) > 5 or np.abs(right_point[0] - points[-1][1][0]) > 5):
    #     if left_point == -1 and right_point == -1:
    #         left_point, right_point = points[-1]

    points.append([left_point, right_point])

    # Draw two points on the image
    if draw is not None:
        if left_point != -1:
            cv2.circle(
                draw, (left_point, interested_line_y), 2, (0, 255 ,0), -1)
        if right_point != -1:
            cv2.circle(
                draw, (right_point, interested_line_y), 2, (0, 255 ,0), -1)
        # cv2.circle(
        #     draw, ((right_point+left_point)//2, interested_line_y), 2, (0, 255 ,0), -1)

    return left_point, right_point, points

def find_left_right_points_2(image, points, draw=None):
    """Find left and right points of lane
    """

    # Consider the position 70% from the top of the image
    interested_line_y_1 = int(HEIGHT * 0.9)
    interested_line_y_2 = int(HEIGHT * 0.7)
    interested_line_y_3 = int(HEIGHT * 0.6)

    if draw is not None:
        cv2.line(draw, (0, interested_line_y_1),
                 (WIDTH, interested_line_y_1), 255, 2)
    interested_line1 = image[interested_line_y_1, :]

    if draw is not None:
        cv2.line(draw, (0, interested_line_y_2),
                 (WIDTH, interested_line_y_2), 255, 2)
    interested_line2 = image[interested_line_y_2, :]

    if draw is not None:
        cv2.line(draw, (0, interested_line_y_3),
                 (WIDTH, interested_line_y_3), 255, 2)
    interested_line3 = image[interested_line_y_3, :]

    # cv2.imshow('', image)
    # cv2.waitKey(0)

    # Detect left/right points
    left_point1 = -1
    right_point1 = -1

    left_point2 = -1
    right_point2 = -1

    left_point3 = -1
    right_point3 = -1

    lane_width = 25
    center = WIDTH // 2

    # Traverse the two sides, find the first non-zero value pixels, and
    # consider them as the position of the left and right lines
    for x in range(center, 0, -1):
        # print(interested_line[x])
        if interested_line1[x] > 0:
            left_point1 = x
            break

    for x in range(center + 1, WIDTH):
        if interested_line1[x] > 0:
            right_point1 = x
            break

    for x in range(center, 0, -1):
        # print(interested_line[x])
        if interested_line2[x] > 0:
            left_point2 = x
            break
    for x in range(center + 1, WIDTH):
        if interested_line2[x] > 0:
            right_point2 = x
            break

    for x in range(center, 0, -1):
        # print(interested_line[x])
        if interested_line3[x] > 0:
            left_point3 = x
            break
    for x in range(center + 1, WIDTH):
        if interested_line3[x] > 0:
            right_point3 = x
            break

    # Predict right point when only see the left point
    if left_point1 != -1 and right_point1 == -1:
        right_point1 = left_point1 + lane_width

    # Predict left point when only see the right point
    if right_point1 != -1 and left_point1 == -1:
        left_point1 = right_point1 - lane_width

    if left_point2 != -1 and right_point2 == -1:
        right_point2 = left_point2 + lane_width

    # Predict left point when only see the right point
    if right_point2 != -1 and left_point2 == -1:
        left_point2 = right_point2 - lane_width

    if left_point3 != -1 and right_point3 == -1:
        right_point3 = left_point3 + lane_width

    # Predict left point when only see the right point
    if right_point3 != -1 and left_point3 == -1:
        left_point3 = right_point3 - lane_width
        
    # if len(points) != 0:
    #     # if (np.abs(left_point - points[-1][0]) > 5 or np.abs(right_point[0] - points[-1][1][0]) > 5):
    #     if left_point == -1 and right_point == -1:
    #         left_point, right_point = points[-1]

    points.append([(left_point1 + right_point1)/2, interested_line_y_1] )
    points.append( [(left_point2 + right_point2)/2, interested_line_y_2])
    points.append([(left_point3 + right_point3)/2, interested_line_y_3] )

    # Draw two points on the image
    if draw is not None:
        if left_point1 != -1:
            cv2.circle(
                draw, (left_point1, interested_line_y_1), 2, (0, 255 ,0), -1)
        if right_point1 != -1:
            cv2.circle(
                draw, (right_point1, interested_line_y_1), 2, (0, 255 ,0), -1)
        if left_point2 != -1:
            cv2.circle(
                draw, (left_point2, interested_line_y_2), 2, (0, 255 ,0), -1)
        if right_point2 != -1:
            cv2.circle(
                draw, (right_point2, interested_line_y_2), 2, (0, 255 ,0), -1)
        if left_point3 != -1:
            cv2.circle(
                draw, (left_point3, interested_line_y_3), 2, (0, 255 ,0), -1)
        if right_point3 != -1:
            cv2.circle(
                draw, (right_point3, interested_line_y_3), 2, (0, 255 ,0), -1)
        
        
        # cv2.circle(
        #     draw, ((right_point+left_point)//2, interested_line_y), 2, (0, 255 ,0), -1)

    return  points


def cal_angle2(points):
    point1 = points[0]
    point2 = points[1]
    point3 = points[2]

    vecto12 = cal_vecto(point1, point2)
    vecto23 = cal_vecto(point2, point3)
    if (cal_distance(point1, point2) * cal_distance(point2, point3)) !=0:
        cos_alpha = (vecto12[0]*vecto23[0] + vecto12[1]*vecto23[1]) / (cal_distance(point1, point2) * cal_distance(point2, point3))
        alpha = np.abs(np.arccos(cos_alpha))
    else:
        alpha = 0
        # cos_alpha = (AB[0]*BC[0] + BC[1]*AB[1]) / (cal_distance(A, B) * cal_distance(B, C))
    if alpha > np.pi/2:
        alpha = np.pi - alpha
    if point3[0] - point2[0] < 0:
        alpha = -alpha
    # print(cos_alpha)
    steering = (alpha/np.pi)
    
    return steering

def calculate_control_signal(left_point, right_point):
    """Calculate speed and steering angle
    """


    # Calculate speed and steering angle
    # The speed is fixed to 50% of the max speed
    # You can try to calculate speed from turning angle
    throttle = 0.3
    steering_angle = 0
    im_center = WIDTH // 2

    # if np.abs(points[-1] - points[-2]) < 8:


    if left_point != -1 and right_point != -1:

        # Calculate the deviation
        center_point = (right_point + left_point) // 2
        center_diff =  center_point - im_center

        # Calculate steering angle
        # You can apply some advanced control algorithm here
        # For examples, PID
        # steering_angle = -pid(center_diff)

        # if np.abs(center_diff) < 5:
        center_diff = -center_diff
        steering_angle = -float(center_diff * 0.01)

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

    print(C, B, A)

    AB = cal_vecto(A, B)
    BC = cal_vecto(B, C)
    if (cal_distance(A, B) * cal_distance(B, C)) !=0:
        cos_alpha = (AB[0]*BC[0] + BC[1]*AB[1]) / (cal_distance(A, B) * cal_distance(B, C))
        alpha = np.abs(np.arccos(cos_alpha))
    
    else:
        alpha = 0
        # cos_alpha = (AB[0]*BC[0] + BC[1]*AB[1]) / (cal_distance(A, B) * cal_distance(B, C))
    if alpha > np.pi/2:
        alpha = np.pi - alpha
    if C[0] - B[0] < 0:
        alpha = -alpha
    # print(cos_alpha)
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
