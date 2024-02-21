#!/bin/bash

# Define base path
BASE_PATH="/zed2i/zed_node"
DIR_PATH="/18ene24"

# Record using rosbag
rosbag record -o ./records$DIR_PATH/zed2_rosbag.bag \
 $BASE_PATH/joint_states \
 $BASE_PATH/atm_press \
 $BASE_PATH/gain_exposure \
 $BASE_PATH/imu/data \
 $BASE_PATH/imu/mag \
 $BASE_PATH/left/camera_info \
 $BASE_PATH/left/image_rect_color/compressed \
 $BASE_PATH/left_cam_imu_transform \
 $BASE_PATH/left_raw/camera_info \
 $BASE_PATH/left_raw/image_raw_color/compressed \
 $BASE_PATH/odom \
 $BASE_PATH/parameter_descriptions \
 $BASE_PATH/parameter_updates \
 $BASE_PATH/path_map \
 $BASE_PATH/path_odom \
 $BASE_PATH/pose \
 $BASE_PATH/pose_with_covariance \
 $BASE_PATH/right/camera_info \
 $BASE_PATH/right/image_rect_color \
 $BASE_PATH/right_raw/camera_info \
 $BASE_PATH/right_raw/image_raw_color \
 $BASE_PATH/temperature/imu \
 $BASE_PATH/temperature/left \
 $BASE_PATH/temperature/right
