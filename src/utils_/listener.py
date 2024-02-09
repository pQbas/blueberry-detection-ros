import rospy
from std_msgs.msg import String

class ROSListener:
    def __init__(self, topic_name):
        self.topic_name = topic_name
        self.subscriber = rospy.Subscriber(self.topic_name, String, self.callback)

    def callback(self, data):
        rospy.loginfo(f"I heard on {self.topic_name}: {data.data}")

    def start(self):
        rospy.init_node('ros_listener', anonymous=True)
        rospy.spin()

if __name__ == '__main__':
    listener = ROSListener("chatter")
    listener.start()
    