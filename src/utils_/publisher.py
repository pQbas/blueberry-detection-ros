#!/usr/bin/env python3
import sys
sys.path.append('/home/pqbas/miniconda3/envs/iot/lib/python3.8/site-packages')

import logging
import sys
from dictionaries import DICTIONARY_DESARROLLO
from geometry_msgs.msg import PoseStamped
from ros_classes import ROSNode
import paho.mqtt.client as mqtt
import time
import rospy
from std_msgs.msg import String
from flask import jsonify
import pickle


logging.basicConfig(level=logging.INFO, force=True)
SIMULATOR = True   # True / False - Verdadero si quiere usar el simulador


# subscriber odometry callback class
class odometry_callback_class():
    def __init__(self):
        self.DICTIONARY_DESARROLLO = None
        pass

    def callback(self, msg):
        self.DICTIONARY_DESARROLLO[0]['info']['x'] = msg.pose.position.x
        self.DICTIONARY_DESARROLLO[0]['info']['y'] = msg.pose.position.y
        self.DICTIONARY_DESARROLLO[0]['info']['z'] = msg.pose.position.z


# create the odometry_callback_
odometry_callback = odometry_callback_class()


def odometry_callback_(msg):
    global DICTIONARY_DESARROLLO
    odometry_callback.DICTIONARY_DESARROLLO = DICTIONARY_DESARROLLO
    odometry_callback.callback(msg)
    DICTIONARY_DESARROLLO = odometry_callback.DICTIONARY_DESARROLLO


from paho.mqtt import client as mqtt_client

import random

broker = 'broker.emqx.io'
port = 1883
topic = "python/mqtt/labinm_tests"
# Generate a Client ID with the publish prefix.
client_id = f'publish-{random.randint(0, 1000)}'
# username = 'emqx'
# password = 'public'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client, msg_):
    msg_count = 0
    while True:
        time.sleep(1)
        msg = msg_
        result = client.publish(topic, msg)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")
        msg_count += 1
        if msg_count > 1:
            break

def mqtt_publisher():
    global DICTIONARY_DESARROLLO
    rate = rospy.Rate(0.5)  # 10 Hz
    while not rospy.is_shutdown():
        client = connect_mqtt()
        client.loop_start()
        publish(client, pickle.dumps(DICTIONARY_DESARROLLO))
        client.loop_stop()
        client.disconnect()

        # dic_str = pickle.dumps(DICTIONARY_DESARROLLO)     
        # mqtt_publish('awdawd')
        print(DICTIONARY_DESARROLLO)
        # pub.publish(msg)
        rate.sleep()

# configuration of subscriber
node = ROSNode("iot_node")
node.create_subscriber("/zed2/zed_node/pose", PoseStamped, odometry_callback_)
pub = rospy.Publisher('python/mqtt/labinm_tests', String, queue_size=0)


if __name__ == '__main__':
    # try:
    mqtt_publisher()
    node.run()

    # except:
    #     print("Node is not running")
    #     exit()
        


