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
import numpy as np
from utils_.conversion_utils import msg2CompresedImage, msg2Image, get_image
from utils_.image_processing_utils import draw_line, crop_center_square, write_text, counter, attach_information_zone
import argparse

def callback(msg):
    img = get_image(msg, TOPIC_NAME)
    img_crop = crop_center_square(img)

    cv2.imshow('Image', img_crop)
    cv2.waitKey(1)

    # if n_arandanos > 10 and n_frame%4 == 0:
    #     saveImage(img_crop)

    return

def createFolder(FOLDER_PATH):    
    return

def saveImage(FOLDER_PATH, img):
    cv2.imwrite(FOLDER_PATH, img)
    return


if __name__ == '__main__': 

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--listener_topic")
    # p = parser.parse_args()
    
    TOPIC_NAME = rospy.get_param('node_configuration/topic_name')
    NODE_NAME = rospy.get_param('node_configuration/node_name')    
    FOLDER_PATH = 'detection_node'

    #createFolder(FOLDER_PATH)

    try:
        rospy.init_node(NODE_NAME, anonymous=True)
        rospy.Subscriber(TOPIC_NAME, CompressedImage, callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass


