#!/usr/bin/env python3
import sys

# my laptop
sys.path.append('/home/pqbas/miniconda3/envs/dl/lib/python3.8/site-packages')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src/detection')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src/detection/object_detection_models/yolov5')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src')


# labinm-jetson
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
import torch
import numpy as np
import pyzed.sl as sl

from object_detection_models.yolo5 import Yolo5
from object_detection_models.yolo8 import Yolo8
from classes.ros_classes import ros_suscriber
from utils_.conversion_utils import msg2CompresedImage, msg2Image
from utils_.image_processing_utils import draw_line, crop_center_square, write_text

ARANDANOS_TOTAL = 0
ARANDANOS_SUMA = 0
ARANDANOS_DETECT = 0
ARANDANOS_CUENTA = 0
TRACKING = True
LIST_0 = []
LIST_1 = []

def callback(msg):
    global n_image
    global ARANDANOS_CUENTA
    global LIST_0
    global LIST_1
    global z
    threshold = 400

    img0 = None
    n_arandanos = 0

    img0 = msg2CompresedImage(msg)  #img0 = ImageCallBack(msg)
    if img0 is None:
        rospy.logwarn('IMG0 is None')
        return
        
    img0 = crop_center_square(img0)
    prediction = detector.predict(img0, conf_thres=0.3, enable_tracking=TRACKING)
    
    
    if prediction is not None:
        img0 = detector.plot_prediction(img0, prediction)
        boxes = prediction[0].boxes.xywh.cpu()
        centers = boxes[:,:2]
        ARANDANOS_DETECT = centers.shape[0]
        img0 = write_text(img0, f'# Detected: {ARANDANOS_DETECT}', (50, 950))

    
    if prediction[0] is not None and prediction is not None and TRACKING == True:
        img0 = draw_line(img0, (200, 200), 'horizontal')
        boxes = prediction[0].boxes.xywh.cpu()
        centers = boxes[:,:2]

        if prediction[0].boxes.id is not None:
            track_ids = prediction[0].boxes.id.int().cpu()
            track_ids = track_ids.reshape(track_ids.shape[0], 1)
            to_count = torch.cat((track_ids, centers),1)

            set_0 = set(LIST_0)
            set_1 = set(LIST_1)
            for (id, x, y) in to_count:
                id = id.item()
                if y < 200:
                    set_0.add(id)       # Adds the id if not already present                        
                    set_1.discard(id)   # Removes the id if present

                elif id in set_0:
                    set_1.add(id)
                
            LIST_0 = list(set_0)
            LIST_1 = list(set_1)

        ARANDANOS_DETECT = centers.shape[0]
        img0 = write_text(img0, f'# Detected: {ARANDANOS_DETECT}', (50, 950))
        ARANDANOS_CUENTA = len(LIST_1)
        img0 = write_text(img0, f'# Counted: {ARANDANOS_CUENTA}', (50, 1000))

    # if prediction is not None:
    #   img0 = detector.plot_prediction(img0, prediction)
    #   img0 = draw_line(img0, (threshold, threshold), 'horizontal')

    count_pub.publish(str(ARANDANOS_CUENTA))
    image_msg = bridge.cv2_to_imgmsg(img0, "bgr8")
    image_pub.publish(image_msg)
    
    n_image += 1
    img0 = cv2.resize(img0, (1000, 1000))
    
    #cv2.imwrite(f'/home/pqbas/catkin_ws/src/blueberry/src/detection/gallery/danper_29Sep23/img{n_image}.png', img0)
       #img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        #img0 = np.array([img0[:,:,2], img0[:,:,1], img0[:,:,0]])
        #print(img0.shape)
    # to show the images
    cv2.imshow('Image', img0)
    cv2.waitKey(1)
    return


def callback_reset(msg):
    global LIST_1, LIST_0
    LIST_1 = []
    LIST_0 = []
    rospy.loginfo(f"Blueberry counting has been ressetted!!!")
    return


if __name__ == '__main__':
    bridge = CvBridge()
    n_image = 0
    img0 = None
    model = 'YOLOV8'
    
    TOPIC_NAME = '/zed2/zed_node/left_raw/image_raw_color/compressed' 
    NODE_NAME = 'detection_node'


    if model == 'YOLOV5':
        detector = Yolo5(weights= '/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/experiment_7/best.pt', #'./yolov5s.pt'
                        data= '', #'/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/experiment_5/custom.yaml', 
                        device='cuda:0')    
    elif model == 'YOLOV8':
        detector = Yolo8(weights='/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/22Sep23/yolov8m_best.engine', #weights='/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/22Sep23/yolov8m_best.pt',
                         device='cuda:0')
    else:
        sys.exit(f"Model not founded")

    try:
        #pub = rospy.Publisher("bbox", Detection2DArray, queue_size=1) --> This is the publisher
        rospy.init_node(NODE_NAME, anonymous=True)
        count_pub = rospy.Publisher('/detection_output/count', String, queue_size=1)
        image_pub = rospy.Publisher('/detection_output/image_topic', Image, queue_size=1)
        
        ros_suscriber('/zed2/zed_node/left_raw/image_raw_color/compressed', CompressedImage, callback)
        ros_suscriber('chatter', String, callback_reset)
        
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
