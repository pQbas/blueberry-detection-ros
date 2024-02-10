#!/usr/bin/env python3
import rospy


class ROSNode:
    def __init__(self, node_name):
        rospy.init_node(node_name, disable_signals=True)
        self.subscribers = {}  # Store subscriber instances in a dictionary

    def create_subscriber(self, topic_name, msg_type, callback):
        # Create a subscriber and store it in the subscribers dictionary
        self.subscribers[topic_name] = rospy.Subscriber(topic_name, msg_type, callback)

    def add_callback_to_subscriber(self, topic_name, new_callback):
        # Retrieve the subscriber instance and add the new callback
        subscriber = self.subscribers.get(topic_name)
        if subscriber:
            subscriber.callback = new_callback

    def run(self):
        rospy.spin()