# Todo
* ~~Publicar el conteo de arandanos en un topico~~
* ~~Reiniciar el número de arandanos que se cuentan~~
* Convertir los pesos de pytorch a tensorrt
* Realizar clasificación de los arandanos que se detectan
* Publicar la deteccion de arandanos en un topico
* Publicar las clases detectadas en un topic
* Utilizar un grafico de barras para graficar las clases que son detectadas

# Research
* ¿Como hacer que una red neuronal de deteccion de objetos, aprenda a detectar y clasificar multiples objetos, a partir de dos redes neuronales una que ya sabe unicamente detectar arandanos y la otra que sabe unicamente clasificar arandanos?
* ¿Cómo es que se utiliza el enfoque probablistico en la detección de objetos?
* ¿En los enfoques probabilisticos, la detección de objetos mejora al considerar la confianza del modelo con respecto a las predicciones que realiza?
* ¿Cómo lograr una mejor detección de objetos pequeños como son los arandanos modificando la arquitectura de la red neuronal empleada?
* ¿Es posible generar datasets mediante el uso de modelos generativos condicionados de tal forma que sea posible generar datasets de deteccion de objetos?

# Steps to run this project

Se utiliza para poder ejecutar el modelo de detección y conteo, ejecutar `roscore` y algun video rosbag del robot antes de ejecutar la siguiente linea.

```bash
$ rosrun blueberry-detection-ros detection-ros.py
```

Se utiliza para reiniciar la cuenta de los arandanos:
```bash
$ rosrun blueberry-detection-ros reset-count.py
```
