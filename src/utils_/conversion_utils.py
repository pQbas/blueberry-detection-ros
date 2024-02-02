import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError

def msg2Image(msg):
    try:
        bridge = CvBridge()
        img0 = bridge.imgmsg_to_cv2(msg, "bgr8")
        rospy.loginfo(rospy.get_caller_id() + " Succeed: Image received" + " Size: " + str(msg.height) + "x" + str(msg.width))
        return img0
    except CvBridgeError:
        rospy.loginfo(rospy.get_caller_id() + " Error: LOL")
        return None
    

def msg2CompresedImage(msg):
    try:
        np_arr = np.fromstring(msg.data, np.uint8)
        img0 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img0
    except CvBridgeError:
        rospy.loginfo(rospy.get_caller_id() + " Error: LOL")
        return None
    
def get_image(msg, TOPIC_NAME):
    img = msg2CompresedImage(msg) if ('compressed' in TOPIC_NAME.split('/')) else msg2Image(msg)
    if img is None: 
        rospy.logwarn('IMG0 is None') 
        return None
    return img

