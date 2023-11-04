#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def publisher():
    rospy.init_node('ros_publisher', anonymous=True)
    pub = rospy.Publisher('chatter', String, queue_size=1)
    rate = rospy.Rate(10)  # 1 Hz (publishes a message every second)

    #while not rospy.is_shutdown():
    message = String()
    message.data = "Hello, ROS!"
    pub.publish(message)
    rospy.loginfo("Published: %s", message.data)
    rate.sleep()


    message = String()
    message.data = "Hello, ROS!"
    pub.publish(message)
    rospy.loginfo("Published: %s", message.data)
    rate.sleep()
    
if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
