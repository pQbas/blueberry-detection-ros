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
import argparse

from object_detection_models.yolo5 import Yolo5
from object_detection_models.yolo8 import Yolo8
from classes.ros_classes import ros_suscriber, ros_publisher
from utils_.conversion_utils import msg2CompresedImage, msg2Image, get_image
from utils_.image_processing_utils import draw_line, crop_center_square, write_text, counter


def callback(msg):
    img = get_image(msg, TOPIC_NAME)
    img_crop = crop_center_square(img)
    prediction = detector.predict(img_crop, conf_thres=0.3, enable_tracking=TRACKING_FLAG)
    img_pred = detector.plot_prediction(img_crop, prediction)

    if TRACKING_FLAG:
        blueberry_counter.update_count(prediction, 500)
        print(blueberry_counter.get_number_counted())

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
    image_pub.publish(img2msg.cv2_to_imgmsg(img_crop, "bgr8"))


if __name__ == '__main__':

    # rosrun blueberry-detection-ros detection-ros.py -model YOLOV5 -sub 'zed2/zed_node/right/image_rect_color/compressed' -show True -track False
    # rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 -sub 'zed2/zed_node/right/image_rect_color/compressed' -show True -track True

    # -------------------------------------------------------------------------------------------
    # Parser arguments: Model  
    # -------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", help = "Object Detection model")
    parser.add_argument("-show", "--show_image", help = "Show image of detection")
    parser.add_argument("-sub", "--subscriber", help = "Suscriber topic, it's the source of the images")
    parser.add_argument("-track", "--tracking_flag", help = "Tracking flag is used to count blueberries")
    args = parser.parse_args()

    if args.model:
        print("Model: % s" % args.model)
        print("Show Image: % s" % args.show_image)
        print("Sub: % s" % args.subscriber)
        print("Track: % s" % args.tracking_flag)
    else:
        sys.exit(f"Model not founded")

    # --------------------------------------------------------------------------------------------
    # Load the model
    # --------------------------------------------------------------------------------------------
    
    MODEL = str(args.model)
    SHOW_IMAGE = eval(args.show_image)
    TRACKING_FLAG = eval(args.tracking_flag)

    if MODEL == 'YOLOV5':
        detector = Yolo5(weights= '/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/experiment_7/best.pt',
                        data= '',
                        device='cuda:0')
    elif MODEL == 'YOLOV8':
        detector = Yolo8(weights='/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/22Sep23/yolov8m_best.pt',
                         device='cuda:0')
    
    if (TRACKING_FLAG == True)  and (MODEL != 'YOLOV8'):   
        sys.exit("Just YoloV8 has tracking methods implemented")
    
    blueberry_counter = counter()

    # -------------------------------------------------------------------------------------------
    # Configure nodes
    # -------------------------------------------------------------------------------------------
    TOPIC_NAME = str(args.subscriber)
    NODE_NAME = 'detection_node'

    try:
        rospy.init_node(NODE_NAME, anonymous=True)
                
        ''' -------------------- Publishers ----------------------------'''
        # Publish the detections
        image_pub = ros_publisher('/detection_output/image_topic', Image, queue_size=1, callback_function=callback_image_publisher) 

        ''' -------------------- Subscribers ----------------------------'''
        # Image from zed2 camera
        if 'compressed' in TOPIC_NAME.split('/'):
            ros_suscriber(TOPIC_NAME, CompressedImage, callback)
        elif ~('compressed' in TOPIC_NAME.split('/')) == True:
            ros_suscriber(TOPIC_NAME, Image, callback)
        else:
            sys.exit(f"TOPIC_NAME not founded")

        # Reset signal of count
        ros_suscriber('chatter', String, callback_reset)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
