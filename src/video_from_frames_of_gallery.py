#Importra librerías
import os
import cv2
#Ubicación de la base de datos
path = './gallery/'
archivos = sorted(os.listdir(path))
img_array = []
 
#Leer imagenes
for x in range (0,len(archivos)):
    nomArchivo = archivos[x]
    dirArchivo = path + str(nomArchivo)
    img = cv2.imread(dirArchivo)
    img_array.append(img)
     
#Dimensiones de los frames alto y ancho
height, width  = img.shape[:2]
 
#Caracteríasticas video
video = cv2.VideoWriter('CVC-08.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height))
 
#Colocar los frames en video
for i in range(0, len(archivos)):
    video.write(img_array[i])
     
#liberar
video.release()