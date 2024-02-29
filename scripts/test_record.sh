#!/bin/bash

# Example:
# ./zed2i_record.sh [base_path] [date] [test_number] [description]
# - base_path: /zed2i/zed_node or /zed2/zed_node
# - date: 21feb24
# - test_number: 1
# - description: counting test, detection test, horizontal, vertical

BASE_PATH=$1
DATE=$2
TEST_NUMBER=$3
DESCRIPTION=$4

# To know the value of variables
echo "base path: $1"
echo "dir path: $2"
echo "test number: $3"
echo "description: $4"

# Create the new dirctory
mkdir -p ~/xavier_ssd/blueberry/records/$DATE/test_$TEST_NUMBER

# Add a description
touch ~/xavier_ssd/blueberry/records/$DATE/test_$TEST_NUMBER/description.txt
echo $DESCRIPTION > ~/xavier_ssd/blueberry/records/$DATE/test_$TEST_NUMBER/description.txt

# Start the rosbag
rosbag record -o ~/xavier_ssd/blueberry/records/$DATE/test_$TEST_NUMBER/testing.bag \
    $BASE_PATH/right_raw/camera_info \
    $BASE_PATH/right/image_rect_color/compressed \
    $BASE_PATH/temperature/imu \
    $BASE_PATH/atm_press \
    /odom \


# $BASE_PATH/left/camera_info \
# $BASE_PATH/left/image_rect_color \
# /detection_output/image_topic


