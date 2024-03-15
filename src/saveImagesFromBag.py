#!/usr/bin/env python3
import sys
sys.path.append('/home/pqbas/miniconda3/envs/dl/lib/python3.8/site-packages')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry/src/detection')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry/src/detection/object_detection_models/yolov5')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src/utils_')


import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose, Detection2DArray
from geometry_msgs.msg import Pose2D, PoseWithCovariance, Pose

import cv2
import os
import numpy as np
from utils_.conversion_utils import msg2CompresedImage, msg2Image, get_image
from utils_.image_processing_utils import draw_line, crop_center_square, write_text, counter, attach_information_zone
import argparse

def callback(msg):
    global n_frame

    img = get_image(msg, TOPIC_NAME)
    img_crop = crop_center_square(img)

    cv2.imshow('Image', img_crop)
    cv2.waitKey(1)

    name = str(n_frame) + '.png'

    if n_frame%4 == 0:
        saveImage(os.path.join(FOLDER_PATH,DATE,TEST_NAME,name), img_crop)

    n_frame += 1
    return

def createFolder(PATH):
    print(os.path.exists(PATH))
    if os.path.exists(PATH) == False:
        os.makedirs(PATH)
        print(f"[WARNING!!!] {PATH} was created!!!")
    else:
        print(f"{PATH} was created before!!!")
    return

def saveImage(FOLDER_PATH, img):
    cv2.imwrite(FOLDER_PATH, img)
    return


if __name__ == '__main__': 


    TOPIC_NAME = rospy.get_param('node_configuration/topic_name')
    NODE_NAME = rospy.get_param('node_configuration/node_name')    
    FOLDER_PATH = rospy.get_param('node_configuration/folder_path')
    DATE = rospy.get_param('node_configuration/date')
    TEST_NAME = rospy.get_param('node_configuration/test_name')

    
    createFolder(os.path.join(FOLDER_PATH,DATE,TEST_NAME))
    
    
    n_frame = 0

    try:
        rospy.init_node(NODE_NAME, anonymous=True)
        rospy.Subscriber(TOPIC_NAME, CompressedImage, callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass


