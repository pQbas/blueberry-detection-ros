# Steps to run this project

### With YoloV5

- Detection

```bash
$ roscore
$ rosbag play zed2_rosbag_2023-09-29-11-55-24.bag
$ rosrun blueberry-detection-ros detection-ros.py -model YOLOV5 \
                                                  -sub 'zed2/zed_node/right/image_rect_color/compressed' \
                                                  -show True \
                                                  -track False
```

### With YoloV8

- Counting in vertical mode

```bash
$ roscore
$ rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 \
                                                  -sub 'zed2/zed_node/right/image_rect_color/compressed' \
                                                  -show True \
                                                  -track True \
                                                  -count_mode vertical \
                                                  -threshold 500
$ rosbag play records/29sepdanper/zed2_rosbag_2023-09-29-11-55-24.bag
```

- Counting in horizontal mode

```bash
$ roscore
$ rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 \
                                                  -sub 'zed2/zed_node/right/image_rect_color/compressed' \
                                                  -show True \
                                                  -track True \
                                                  -count_mode horizontal \
                                                  -threshold 500
$ rosbag play records/29sepdanper/zed2_rosbag_2023-09-29-11-58-00.bag
```


# Robot connection

1. SSH conection:

```bash
ssh ubuntu@192.168.0.40
password: pi123456
ssh labinm-jetson@192.168.0.10
password: rpgdini100
```

2. ZED2i:

```bash
roslaunch zed_wrapper zed2i.launch
```

3. blueberry detector activation:

```bash
rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 \
                                                  -sub '/zed2i/zed_node/left/image_rect_color' \
                                                  -show False \
                                                  -track False \
                                                  -count_mode horizontal \
                                                  -threshold 500
```

4. execute rviz to visualize:
```
rviz
```


# Detection Launch

The content of the file: `src/detection.launch`

```yaml
<launch>
  
	<include 
		file="$(find zed_wrapper)/launch/zed2i.launch" 
	/>

	<node 
		pkg="blueberry-detection-ros"
		type="detection-ros.py"
		name="detection_node"  
		output="screen"
	/>

</launch>
```






