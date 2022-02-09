import pandas as pd
import os
import numpy as np
import cv2

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
max_frames = 120
count_generated_imgs=0
joints_order = [1,2,3,26,27,28,29,28,27,30,31,30,27,26,3,2,4,5,6,7,8,9,8,7,10,7,6,5,4,2,11,12,13,14,
                15,16,15,14,17,14,13,12,11,2,1]
img = np.zeros((max_frames, len(joints_order),3), dtype=np.uint8)
for nombre_directorio, dirs, ficheros in os.walk(skeleton_path, topdown=False):
    for nombre_fichero in ficheros:
        ske=os.path.join(nombre_directorio,nombre_fichero)
        # print(os.path.join(nombre_directorio,nombre_fichero))
        frame = pd.read_csv(ske, sep="\t", header=0)
        frame = pd.DataFrame(frame)
        joints={}
        #     con esto tenemos que ir leyendo los esqueletos y crear la imagen con todos.
        #     Tendríamos que poner el límite de cuantos frames vamos a tener para generar cada imagen
        for j, line in frame.iterrows():
            # print(line)
            # print(normalize(line[3],-200, 1000, 255, 0))
            # print(normalize(line[4],-1000, 1000, 255, 0))
            # print(normalize(line[5],1200, 2200, 255, 0))

            joints[int(line[1])]=[normalize(line[3],-200, 1000, 255, 0),normalize(line[4],-1000, 1000, 255, 0),
                                  normalize(line[5],1200, 2200, 255, 0)]

        # print(joints)
        for idx in range(len(joints_order)):
            # print(jo)
            # print(joints[joints_order[idx]][0])
            img[i-1,idx,0]= joints[joints_order[idx]][0]
            img[i-1,idx,1]= joints[joints_order[idx]][1]
            img[i-1,idx,2]= joints[joints_order[idx]][2]


        #     # aquí tenemos que saber el orden en el que usar el arbol del esqueleto. Pero creo que la mejor forma de trabajar
        #     # es crear un diccionario que guarde como clave el id del joint y luego el resto de datos. Con esto es más facil
        #     # recorrerlos y luego poder generar la imagen. Para la imagen tenemos que saber como normalizar las distancias.
        #     # además de saber como crear la imagen, en el artículo lo ponen en escala de grises pero igual es mejor hacerlo
        #     # en rgb para conservar la información de cada componente.
        # # Aquí creamos el recorrido del esqueleto y luego generamos la imagen
        #
        if i< max_frames and (count_generated_imgs<len(ficheros) and i<len(ficheros)):
            # Aqui deberiamos añadir tambien el caso en el que se acaben los esqueletos que leer. Esos tambien hay
            # que añadirlos en la imagen
            i+=1
        else:
            generated_image="from_"+ str(count_generated_imgs)+"_to_"+ (str(count_generated_imgs+max_frames)+".jpg")
            count_generated_imgs+=max_frames
            i=1
            cv2.imwrite(generated_image, img)




















