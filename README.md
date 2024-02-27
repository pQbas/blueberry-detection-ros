<div align="center">
    <h1>Blueberry-Detection-ROS</h1>

  <p align="center">
    <a href="here_is_a_demo_video"><img alt="Blueberry Detection ROS" src="gallery/image-demo.png"></a>
  </p>

</div>

This is a repository of a computer vision system to detect and classify blueberries in agroindustrial enviroments based on YoloV5/YoloV8 techniques, the model run over a Jetson Xavier.

## Installation & Testing:

Clone and install all requirements:

```bash
git clone https://github.com/pQbas/blueberry-detection-ros.git
cd blueberry-detection-ros
pip install -r requirements.txt
```

Download the weights and records:

```bash
./weights/download_weights.sh
./records/download_test_records.sh
```

Run YoloV8 for blueberry **counting** using ROS framework:

### Testing locally

Records generally use zed2 topics:

```bash
# horizontal-mode
rosbag play records/zed2_rosbag_2023-09-29-11-55-24.bag -l
rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 \
                                                  -sub 'zed2/zed_node/right/image_rect_color/compressed' \
                                                  -show True \
                                                  -track True \
                                                  -count_mode vertical \
                                                  -threshold 500 \
                                                  -direction top2down \
                                                  -weights 'weights/yolov8m_best.pt'

# vertical-mode
rosbag play records/zed2_rosbag_2023-09-29-12-10-05.bag -l
rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 \
                                                  -sub 'zed2/zed_node/right/image_rect_color/compressed' \
                                                  -show True \
                                                  -track True \
                                                  -count_mode horizontal \
                                                  -threshold 500 \
                                                  -direction right2left \
                                                  -weights 'weights/yolov8m_best.pt'
```

### Testing on Robot

Records generally use zed2i topics:

```bash
# run zed2i camera
roslaunch zed_wrapper zed2i.launch

# run detection node
rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 \
                                                  -sub '/zed2i/zed_node/left/image_rect_color' \
                                                  -show True \
                                                  -track True \
                                                  -count_mode horizontal \
                                                  -threshold 320 \
                                                  -direction left2right \
                                                  -weights 'weights/yolov8m_best.pt'

rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 \
                                                  -sub '/zed2i/zed_node/left/image_rect_color' \
                                                  -show True \
                                                  -track True \
                                                  -count_mode vertical \
                                                  -threshold 320 \
                                                  -direction top2down \
                                                  -weights 'weights/yolov8m_best.pt'

```

## Procedimiento

Start the detection and counting system:

```bash
# run zed2i camera
roslaunch zed_wrapper zed2i.launch  

# run detection node
cd catkin_ws/src/blueberry-detection-ros/
rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 \
                                                  -sub '/zed2i/zed_node/left/image_rect_color' \
                                                  -show False \
                                                  -track True \
                                                  -count_mode horizontal \
                                                  -threshold 500 \
                                                  -direction left2right \
                                                  -weights 'weights/yolov8m_best.pt'
```

Save records `./scripts/test_record.sh [base_path] [today_date] [n_test] [description]`:

```bash
./scripts/test_record.sh /zed2i/zed_node 22feb24 1 "horizontal"
```

### To-do:
- [ ] Limpiar el ssd para poder realizar grabaciones.
- [ ] Definir las pruebas a realizar.
- [ ] Realizar una práctica de las pruebas a realizar con el robot en campo.

### Publications:

1. Artificial vision strategy for Ripeness assessment of Blueberries on Images taken during Pre-harvest stage in Agroindustrial Environments using Deep Learning Techniques. INTERCON2023. (https://ieeexplore.ieee.org/document/10326058)
2. Detection and Classification of ventura-blueberries in five levels of ripeness from images taken during pre-harvest stage using Deep Learning techniques. ANDESCON2022. (https://ieeexplore.ieee.org/document/9989578)



<!-- Run YoloV5/YoloV8 for blueberry **detection** using ROS framework:

```bash
roscore
rosbag play records/zed2_rosbag_2023-09-29-12-10-05.bag
rosrun blueberry-detection-ros detection-ros.py -model YOLOV5 \
                                                  -sub 'zed2/zed_node/right/image_rect_color/compressed' \
                                                  -show True \
                                                  -track False
```
 -->


<!-- rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 \
                                                  -sub '/zed2i/zed_node/left/image_rect_color' \
                                                  -show False \
                                                  -track True \
                                                  -count_mode horizontal \
                                                  -threshold 500 \
                                                  -direction right2left \
                                                  -weights 'weights/yolov8m_best.pt' -->



<!-- 
# Robot connection

1. SSH conection:

```bash
ssh ubuntu@192.168.0.40
pi123456
ssh labinm-jetson@192.168.0.10
rpgdini100
```

1. ZED2i:

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





 -->
