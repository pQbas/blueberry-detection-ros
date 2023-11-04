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
TRACKING = False
LIST_0 = []
LIST_1 = []


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
    global n_image
    global ARANDANOS_CUENTA
    global LIST_0
    global LIST_1
    global z
    threshold = 400

    img0 = None
    n_arandanos = 0

    img0 = CompresedImageCallBack(msg)  #img0 = ImageCallBack(msg)
    #img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    if img0 is not None:
        img0 = crop_center_square(img0)
        #img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        #img0 = np.array([img0[:,:,2], img0[:,:,1], img0[:,:,0]])
        #print(img0.shape)
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

if __name__ == '__main__':
    # A set of things that are important
    bridge = CvBridge()
    n_image = 0
    img0 = None
    model = 'YOLOV8'
    
    # TOPIC_NAME, NODE_NAME variables, important.
    TOPIC_NAME = '/zed2/zed_node/left_raw/image_raw_color/compressed'#'/zed2/zed_node/right/image_rect_color/compressed' #'/zed2i/zed_node/right/image_rect_color/compressed' #  
    #TOPIC_NAME = '/zed2i/zed_node/rgb_raw/image_raw_color' #
    NODE_NAME = 'detection_node'

    # Defining the object detector
    if model == 'YOLOV5':
        detector = Yolo5(weights= '/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/experiment_7/best.pt', #'./yolov5s.pt'
                        data= '', #'/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/experiment_5/custom.yaml', 
                        device='cuda:0')
    
    elif model == 'YOLOV8':
        detector = Yolo8(weights='/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/22Sep23/yolov8m_best.pt',
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
    
    # object_detected = Detection2D()
    # We still don't use this tracker...... 
    # I think we need to improve object detection use more trackers to perform a best comparison
    # Indeed, the true is that the data obtained from DAMPER is what matters, the rest is like a
    # artifacts to perform some task
    #tracker = DeepOCSORT(
    #    model_weights=Path('/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/mobilenetv2_x1_4_dukemtmcreid.pt'),  # which ReID model to use, when applicable
    #    device='cuda:0',  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
    #    fp16=True,  # wether to run the ReID model with half precision or not
    #    det_thresh=0.2  # minimum valid detection confidence
    #)
    
