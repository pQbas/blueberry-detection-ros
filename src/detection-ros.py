#!/usr/bin/env python3
import sys

''' pqbas laptop '''
sys.path.append('/home/pqbas/miniconda3/envs/dl/lib/python3.8/site-packages')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src/detection')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src/detection/object_detection_models/yolov5')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src')

''' labinm_jetson '''
sys.path.append('/home/labinm-jetson/catkin_ws/src/blueberry-detection-ros/src/detection')
sys.path.append('/home/labinm-jetson/pqbas/catkin_ws/src/blueberry-detection-ros/src/detection/object_detection_models/yolov5')
sys.path.append('/home/labinm-jetson/pqbas/catkin_ws/src/blueberry-detection-ros/src')


import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose, Detection2DArray
from geometry_msgs.msg import Pose2D, PoseWithCovariance, Pose
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import argparse
from object_detection_models.yolo5 import Yolo5
from object_detection_models.yolo8 import Yolo8
from classes.ros_classes import ros_suscriber, ros_publisher
from utils_.conversion_utils import msg2CompresedImage, msg2Image, get_image
from utils_.image_processing_utils import draw_line, crop_center_square, write_text, counter, attach_information_zone
import os


def callback(msg):
    global img_pred

    img = get_image(msg, TOPIC_NAME)
    img_crop = crop_center_square(img)
    prediction = detector.predict(img_crop, conf_thres=0.3, enable_tracking=TRACKING_FLAG)
    img_pred = detector.plot_prediction(img_crop, prediction)

    if TRACKING_FLAG:
        blueberry_counter.update_count(prediction)
        img_pred = blueberry_counter.plot_line_threshold(img_pred)
        number_blueberries = blueberry_counter.get_number_counted()['counted']

    # ------ DESCRIPTION SHOWED IN PLOTT ----------
    img_pred = attach_information_zone(img_pred)
    write_text(img_pred, 'Detected: ', position=(20, 50), scale_font=1, thick=2, color=(255, 255, 255))
    write_text(img_pred, str(prediction[0].boxes.xywh.shape[0]), position=(20, 150), scale_font=3, thick=2, color=(255, 255, 255))

    write_text(img_pred, 'Counted: ', position=(20, 200), scale_font=1, thick=2, color=(255, 255, 255))
    write_text(img_pred, str(number_blueberries), position=(20, 300), scale_font=3, thick=2, color=(255, 255, 255))

    write_text(img_pred, 'count mode: ', position=(20, 350), scale_font=1, thick=2, color=(255, 255, 255))
    write_text(img_pred, str(COUNT_MODE), position=(20, 400 + 20), scale_font=2, thick=2, color=(255, 255, 255))

    write_text(img_pred, 'direction: ', position=(20, 500), scale_font=1, thick=2, color=(255, 255, 255))
    write_text(img_pred, str(DIRECTION), position=(20, 550 + 20), scale_font=2, thick=2, color=(255, 255, 255))
    
    write_text(img_pred, 'threshold: ', position=(20, 650), scale_font=1, thick=2, color=(255, 255, 255))
    write_text(img_pred, str(THRESHOLD_TRACK), position=(20, 700 + 20), scale_font=2, thick=2, color=(255, 255, 255))
    
    write_text(img_pred, 'topic name: ', position=(20, 800), scale_font=1, thick=2, color=(255, 255, 255))
    write_text(img_pred, str(TOPIC_NAME), position=(20, 850), scale_font=1, thick=2, color=(255, 255, 255))


    if SHOW_IMAGE and (prediction is not None):
        cv2.imshow('Image', img_pred)
        cv2.waitKey(1)

    return


def callback_reset(msg):
    global LIST_1, LIST_0
    LIST_1 = []
    LIST_0 = []
    rospy.loginfo(f"Blueberry counting has been ressetted!!!")
    return


img2msg = CvBridge()
def callback_image_publisher(img_crop):
    image_pub.publish(img_crop)
    return


if __name__ == '__main__':

    # -------------------------------------------------------------------------------------------
    # Parser arguments: Model  
    # -------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", help = "Object Detection model")
    parser.add_argument("-show", "--show_image", help = "Show image of detection")
    parser.add_argument("-sub", "--subscriber", help = "Suscriber topic, it's the source of the images")
    parser.add_argument("-track", "--tracking_flag", help = "Tracking flag is used to count blueberries")
    parser.add_argument("-count_mode", "--count_mode", help = "Counting mode is 'Horizontal' or 'Vertical'")
    parser.add_argument("-threshold_track", "--threshold_track", help='threshold of the tracker')
    parser.add_argument("-direction","--direction", help=' direction is: "right2left","left2right","up2down","down2top" ' )
    parser.add_argument("-weights","--weights_path", help='weights path')
    args = parser.parse_args()

    if args.model:
        print("Model: % s" % args.model)
        print("Show Image: % s" % args.show_image)
        print("Sub: % s" % args.subscriber)
        print("Track: % s" % args.tracking_flag)
    else:
        sys.exit(f"Model not founded")

    MODEL = str(args.model)
    SHOW_IMAGE = eval(args.show_image)
    TRACKING_FLAG = eval(args.tracking_flag)
 
    COUNT_MODE = str(args.count_mode)
    THRESHOLD_TRACK = int(args.threshold_track)
    DIRECTION = str(args.direction)
 
    TOPIC_NAME = str(args.subscriber)
    NODE_NAME = 'detection_node'

    WEIGHTS_PATH = str(args.weights_path)

    # --------------------------------------------------------------------------------------------
    # Load the model
    # --------------------------------------------------------------------------------------------
    
    if MODEL == 'YOLOV5':
        
        detector = Yolo5(weights=WEIGHTS_PATH,
                        data='',
                        device='cuda:0')
    elif MODEL == 'YOLOV8':
        detector = Yolo8(weights=WEIGHTS_PATH,
                         device='cuda:0')
    
    if (TRACKING_FLAG == True)  and (MODEL != 'YOLOV8'):   
        sys.exit("Just YoloV8 has tracking methods implemented")
    

    blueberry_counter = counter(count_mode=COUNT_MODE, threshold_track=THRESHOLD_TRACK, direction=DIRECTION)

    # -------------------------------------------------------------------------------------------
    # Configure nodes
    # -------------------------------------------------------------------------------------------
    img_pred = np.zeros([600,600,3], dtype=np.uint8)

    try:
        rospy.init_node(NODE_NAME, anonymous=True)

        ''' -------------------- Publishers ----------------------------'''
        # Publish the detections
        image_pub = rospy.Publisher('/detection_output/image_topic', Image, queue_size=1)
        ''' -------------------- Subscribers ----------------------------'''
        # Image from zed2 camera
        if 'compressed' in TOPIC_NAME.split('/'):
            ros_suscriber(TOPIC_NAME, CompressedImage, callback)
        elif ~('compressed' in TOPIC_NAME.split('/')):
            ros_suscriber(TOPIC_NAME, Image, callback)
        else:
            sys.exit(f"TOPIC_NAME not founded")
        
        ''' -------------------- Running Publisher -----------------------------'''
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            image_pub.publish(img2msg.cv2_to_imgmsg(img_pred, "bgr8"))
            r.sleep()
            

    except rospy.ROSInterruptException:
        pass
