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


class ros_publisher:
    def __init__(self, topic_name, data_type=None, queue_size=1, callback_function=None):
        self.topic_name = topic_name
        
        if callback_function == None:
            self.publisher = rospy.Publisher(self.topic_name, data_type, self.callback, queue_size=queue_size)
        else:
            self.publisher = rospy.Publisher(self.topic_name, data_type, callback_function,  queue_size=queue_size)

            
    def callback(self, data):
        rospy.loginfo(f"I heard on {self.topic_name}: {data.data}")

    def publish(self, data):
        self.publisher.publish(data)
        


if __name__ == '__main__':
    listener = ros_suscriber("chatter")
