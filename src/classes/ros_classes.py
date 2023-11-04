#!/usr/bin/env python

import rospy
from std_msgs.msg import String

class ros_suscriber:
    def __init__(self, topic_name, data_type=String, callback_function=None):
        self.topic_name = topic_name
        
        if callback_function == None:
            self.subscriber = rospy.Subscriber(self.topic_name, data_type, self.callback)
        else:
            self.subscriber = rospy.Subscriber(self.topic_name, data_type, callback_function)

            
    def callback(self, data):
        rospy.loginfo(f"I heard on {self.topic_name}: {data.data}")

if __name__ == '__main__':
    listener = ros_suscriber("chatter")
