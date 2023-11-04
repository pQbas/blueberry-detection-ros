import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError

def msg2Image(msg):
    try:
        bridge = CvBridge()
        img0 = bridge.imgmsg_to_cv2(msg.data, "bgr8")
        rospy.loginfo(rospy.get_caller_id() + " Succeed: Image received" + " Size: " + str(msg.height) + "x" + str(msg.width))
        return img0
    except CvBridgeError:
        rospy.loginfo(rospy.get_caller_id() + " Error: LOL")
        return
    

def msg2CompresedImage(msg):
    try:
        np_arr = np.fromstring(msg.data, np.uint8)
        img0 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img0
    except CvBridgeError:
        rospy.loginfo(rospy.get_caller_id() + " Error: LOL")
        return
    