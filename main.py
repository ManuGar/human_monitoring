import pandas as pd
import os
import numpy as np
import cv2

# Función para normalizar los valores de las posiciones de los joint. Tienen que estar entre 0 y 255
# para poder guardarlos en los píxeles de la imagen
def normalize(value: float, lower_bound: float, higher_bound: float, max_value: int, min_value: int) -> float:
    if value > higher_bound:
        ret_value = max_value
    elif value < lower_bound:
        ret_value = min_value
    else:
        ret_value = (max_value * ((value - lower_bound) / (higher_bound - lower_bound)))  # estava com cast de int()
    return ret_value

# skeleton_path = 'IDU001V001_20220119_155742'
skeleton_path = 'ske'

i = 1
# Límite de frames que se van a incluir en cada imagen
max_frames = 120
count_generated_imgs=0

# En este vector guardamos el orden de los joint para luego poder recorrerlo en el orden que queremos
joints_order = [1,2,3,26,27,28,29,28,27,30,31,30,27,26,3,2,4,5,6,7,8,9,8,7,10,7,6,5,4,2,11,12,13,14,
                15,16,15,14,17,14,13,12,11,2,1]
img = np.zeros((max_frames, len(joints_order),3), dtype=np.uint8)

# Vamos recorriendo las carpetas y si se cambia de carpeta, dejaremos el resto de la imagen en negro y se empezará
# la siguiente
for nombre_directorio, dirs, ficheros in os.walk(skeleton_path, topdown=False):
    print(nombre_directorio)
    for nombre_fichero in ficheros:
        ske=os.path.join(nombre_directorio,nombre_fichero)
        # print(os.path.join(nombre_directorio,nombre_fichero))
        frame = pd.read_csv(ske, sep="\t", header=0)
        frame = pd.DataFrame(frame)
        joints={}
        # Con esto tenemos que ir leyendo los frames para obtener de cada uno la información de los esqueletos y
        # generar una fila de la imagen para cada frame que leamos.


        # Aquí guardamos los joints y los datos que queremos en un diccionario para luego generar la imagen
        for j, line in frame.iterrows():
            # print(line)
            # print(normalize(line[3],-200, 1000, 255, 0))
            # print(normalize(line[4],-1000, 1000, 255, 0))
            # print(normalize(line[5],1200, 2200, 255, 0))
            joints[int(line[1])]=[normalize(line[3],-200, 1000, 255, 0),normalize(line[4],-1000, 1000, 255, 0),
                                  normalize(line[5],1200, 2200, 255, 0)]

        # print(joints)
        # Guardamos el contenido de las componentes de los diferentes joints en las coordenadas de la imagen
        # para cada frame generamos una fila en la imagen
        for idx in range(len(joints_order)):
            # print(jo)
            # print(joints[joints_order[idx]][0])
            img[i-1,idx,0]= joints[joints_order[idx]][0]
            img[i-1,idx,1]= joints[joints_order[idx]][1]
            img[i-1,idx,2]= joints[joints_order[idx]][2]

        # Aquí vamos a escribir la matriz con los datos a la imagen para guardarla. Primero comprobamos que podamos
        # seguir guardando la matriz con los datos de la imagen que generamos. Si hemos llegado al límite de frames o
        # al final del fichero, lo que haremos es escribirla en la imagen.
        if i< max_frames and (count_generated_imgs<len(ficheros) and i<len(ficheros)):
            i+=1
        else:
            generated_image="from_"+ str(count_generated_imgs)+"_to_"+ (str(count_generated_imgs+max_frames)+".jpg")
            count_generated_imgs+=max_frames
            i=1
            cv2.imwrite(generated_image, img)




















