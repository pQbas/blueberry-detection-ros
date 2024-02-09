# Steps to run this project

Se utiliza para poder ejecutar el modelo de detección y conteo, ejecutar `roscore` y algun video rosbag del robot antes de ejecutar la siguiente linea.

```bash
$ roscore
$ rosbag play zed2_rosbag_2023-09-29-11-55-24.bag -l
$ rosrun blueberry-detection-ros detection-ros.py -model YOLOV5 -sub 'zed2/zed_node/right/image_rect_color/compressed' -show True -track False
```

```bash
$ rosbag play zed2_rosbag_2023-09-29-11-55-24.bag -l
$ rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 -sub 'zed2/zed_node/right/image_rect_color/compressed' -show True -track True -count_mode vertical
```

```bash
$ rosbag play zed2_rosbag_2023-09-29-11-58-00.bag -l
$ rosrun blueberry-detection-ros detection-ros.py -model YOLOV8 -sub 'zed2/zed_node/right/image_rect_color/compressed' -show True -track True -count_mode horizontal -threshold 500
```

Se utiliza para reiniciar la cuenta de los arandanos:

```bash
$ rosrun blueberry-detection-ros reset-count.py
```

# Todo
* ~~Publicar el conteo de arandanos en un topico~~
* ~~Reiniciar el número de arandanos que se cuentan~~
* ~~Convertir los pesos de pytorch a tensorrt~~
* Publicar la deteccion de arandanos en un topico
* Realizar clasificación de los arandanos que se detectan
* Publicar las clases detectadas en un topic
* Utilizar un grafico de barras para graficar las clases que son detectadas
