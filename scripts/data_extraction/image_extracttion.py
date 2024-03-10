#!/usr/bin/env python3
import sys
sys.path.append('/home/pqbas/miniconda3/envs/dl/lib/python3.8/site-packages')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry/src/detection')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry/src/detection/object_detection_models/yolov5')


import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose, Detection2DArray
from geometry_msgs.msg import Pose2D, PoseWithCovariance, Pose
import rospy
import pyzed.sl as sl
import cv2
import numpy as np
import torch
from object_detection_models.yolo5 import Yolo5
from object_detection_models.yolo8 import Yolo8
from callbacks import CompresedImageCallBack, ImageCallBack
from common_functions import write_text


ARANDANOS_TOTAL = 0
ARANDANOS_SUMA = 0
ARANDANOS_DETECT = 0
ARANDANOS_CUENTA = 0
LIST_0 = []
LIST_1 = []
num_image = 0
n_frame = 0


def draw_line(image, position, orientation):

    h,w,c = image.shape
    color = (255,0,0)
    thickness = 2
    
    if orientation == 'vertical':
        start_point = (position[0],0)
        end_point = (position[0], h)

    elif orientation == 'horizontal':
        start_point = (0, position[1])
        end_point = (w, position[1])

    image = cv2.line(image, start_point, end_point, color, thickness)
    return image

def crop_center_square(image):
    h, w, _ = image.shape
    size = min(h, w)
    x_start = (w - size) // 2
    x_end = x_start + size
    y_start = (h - size) // 2
    y_end = y_start + size
    cropped = image[y_start:y_end, x_start:x_end]
    return cropped

def callback(msg):
    global num_image, n_frame, ARANDANOS_CUENTA, LIST_0, LIST_1, z
    threshold = 400

    img0 = None
    n_arandanos = 0

    img0 = CompresedImageCallBack(msg)  #img0 = ImageCallBack(msg)
    img0_copy = img0.copy()
    #img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    if img0_copy is not None:
        img0_copy = crop_center_square(img0_copy)
        prediction = detector.predict(img0_copy, conf_thres=0.3, enable_tracking=False)
        
        n_arandanos = prediction.shape[0]
        if prediction is not None:
           img0_copy = detector.plot_prediction(img0_copy, prediction)
           img0_copy = draw_line(img0_copy, (threshold, threshold), 'horizontal')    

    img0_copy = cv2.resize(img0_copy, (800, 800))

    cv2.imshow('Image', img0_copy)
    cv2.waitKey(1)

    if n_arandanos > 10 and n_frame%4 == 0:
        cv2.imwrite(f'/home/pqbas/catkin_ws/src/blueberry/src/detection/gallery/danper_15Sep23/img0_{num_image}.png', img0)
        num_image += 1
    n_frame += 1

    return

if __name__ == '__main__':
    # A set of things that are important
    bridge = CvBridge()
    n_image = 0
    img0 = None
    model = 'YOLOV5'
    
    # TOPIC_NAME, NODE_NAME variables, important.
    TOPIC_NAME = '/zed2i/zed_node/right/image_rect_color/compressed' #'/zed2i/zed_node/right/image_rect_color/compressed' #  '/zed2i/zed_node/rgb_raw/image_raw_color' #
    NODE_NAME = 'detection_node'

    # Defining the object detector
    if model == 'YOLOV5':
        detector = Yolo5(weights= '/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/experiment_7/best.pt', #'./yolov5s.pt'
                        data= '', #'/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/experiment_5/custom.yaml', 
                        device='cuda:0')
    
    elif model == 'YOLOV8':
        detector = Yolo8(weights='/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/yolov8m_1kepochs/weights/best.pt',
                         device='cuda:0')
    else:
        sys.exit(f"Model not founded")
    try:
        #pub = rospy.Publisher("bbox", Detection2DArray, queue_size=1) --> This is the publisher
        rospy.init_node(NODE_NAME, anonymous=True) # Node intialization with {NODE_NAME}
        count_pub = rospy.Publisher('/detection_output/count', String, queue_size=1)
        image_pub = rospy.Publisher('/detection_output/image_topic', Image, queue_size=1)

        #rospy.Subscriber(TOPIC_NAME, Image, callback)
        rospy.Subscriber(TOPIC_NAME, CompressedImage, callback)  # Suscription to ZED2i camera
        rospy.spin() # I don't know what does it, but I know that is necesary
    except rospy.ROSInterruptException:
        pass
