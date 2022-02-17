import argparse
import pandas as pd
import os
import numpy as np
import cv2
import annotation_images
from imutils import paths

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


def generate_images(skeleton_path, max_frames=68, stride=20):

    i = 1
    # Límite de frames que se van a incluir en cada imagen
    # max_frames = 128 #Esto es por el resultado que nos ha dado la media y la mediana
    count_generated_frames=0

    # En este vector guardamos el orden de los joint para luego poder recorrerlo en el orden que queremos
    joints_order = [1,2,3,26,27,28,29,28,27,30,31,30,27,26,3,2,4,5,6,7,8,9,8,7,10,7,6,5,4,2,11,12,13,14,
                    15,16,15,14,17,14,13,12,11,2,1]
    img = np.zeros((max_frames, len(joints_order),3), dtype=np.uint8)

    # Vamos recorriendo las carpetas y si se cambia de carpeta, dejaremos el resto de la imagen en negro y se empezará
    # la siguiente
    l=0 #contador para recorrer los ficheros con un while y poder hacer bien el stride
    for nombre_directorio, dirs, ficheros in os.walk(skeleton_path):
        # print(nombre_directorio)
        # for nombre_fichero in ficheros:
        print(nombre_directorio)
        print(dirs)
        print("PRUEBAAAAAAAAAAAAAAAAAAAAAA")
        count_generated_frames = 0
        while l < len(ficheros) and count_generated_frames <= len(ficheros):
            nombre_fichero = ficheros[l]
            ske=os.path.join(nombre_directorio,nombre_fichero)
            # print(os.path.join(nombre_directorio,nombre_fichero))
            frame = pd.read_csv(ske, sep="\t", header=0)
            frame = pd.DataFrame(frame)
            joints={}
            # Con esto tenemos que ir leyendo los frames para obtener de cada uno la información de los esqueletos y
            # generar una fila de la imagen para cada frame que leamos.

            # Aquí guardamos los joints y los datos que queremos en un diccionario para luego generar la imagen
            for j, line in frame.iterrows():
                # Los valores máximos y mínimos han sido seleccionados mirando los valores que obtienen los esqueletos
                # que ibamos a estudiar (espero que esos valores se conserven también en los otros vídeos)
                joints[int(line[1])]=[normalize(line[3],-200, 1000, 255, 0),normalize(line[4],-1000, 1000, 255, 0),
                                      normalize(line[5],1200, 2200, 255, 0)]

            # print(joints)
            # Guardamos el contenido de las componentes de los diferentes joints en las coordenadas de la imagen
            # para cada frame generamos una fila en la imagen
            for idx in range(len(joints_order)):
                # print(joints[joints_order[idx]][0])
                img[i-1,idx,0]= joints[joints_order[idx]][0]
                img[i-1,idx,1]= joints[joints_order[idx]][1]
                img[i-1,idx,2]= joints[joints_order[idx]][2]

            # Aquí vamos a escribir la matriz con los datos a la imagen para guardarla. Primero comprobamos que podamos
            # seguir guardando la matriz con los datos de la imagen que generamos. Si hemos llegado al límite de frames o
            # al final del fichero, lo que haremos es escribirla en la imagen.
            if i < max_frames and (l+1 < len(ficheros) and i < len(ficheros)):
                i += 1
                l += 1
            else:
                # Guardamos las imágenes generadas en una carpeta para que esté organizada junto al vídeo que hacen
                # referencia y poder saber fácilmente la clase de cada fila de las imágenes.
                if not os.path.exists(os.path.join(nombre_directorio,"..","..","images")):
                    os.mkdir(os.path.join(nombre_directorio,"..","..","images"))
                generated_image=os.path.join(nombre_directorio,"..","..","images","from_" + str(count_generated_frames) + "_to_" + (str(count_generated_frames + max_frames) ))

                # Aquí si cambiamos de problema o modificamos las clases que queremos estudiar habrá que cambiar ese 12
                # que hace referencia al número de clases del problema que estamos tratando






                # probar esta parte con todo lo de las rutas y hacer pruebas en donde se guarda y dejarlo
                #  todo organizado para luego poder entrenarlo
                id_video = nombre_directorio.split(os.path.sep)[1].split("_")[0]
                # Hay que revisar y comprobar que esto valga para todos los casos. Ponerlo de forma que sirva siempre.
                # De momento solo vale si las carpetas que contienen los esqueletos están dentro de una carpeta "padre"
                for ann_files in os.listdir("annot_renamed"):
                    id_ann_file = ann_files.split("_")[0]

                    if id_ann_file==id_video:
                        annotation_path = os.path.join("annot_renamed",ann_files)
                        break

                # annotation_path = os.path.join("annot_renamed",nombre_directorio)
                output_path = os.path.join(nombre_directorio,"..","..", "annotation_images_labelSmoothing.csv")
                annotation_images.annotation_images_labelsmoothing(count_generated_frames,count_generated_frames+max_frames,12,annotation_path,output_path)


                count_generated_frames += max_frames-stride
                i = 1
                cv2.imwrite(generated_image + ".jpg", img)
                cv2.imwrite(generated_image + "X.jpg", img[:,:,0])
                cv2.imwrite(generated_image + "Y.jpg", img[:,:,1])
                cv2.imwrite(generated_image + "Z.jpg", img[:,:,2])
                img = np.zeros((max_frames, len(joints_order), 3), dtype=np.uint8)
                l -= stride
                l += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skeleton_path', type=str,
                        help='path of the skeleton files')
    parser.add_argument('--max_frames', type=int, default=68,
                        help='number of frames to store in each image')
    parser.add_argument('--stride', type=int, default=20,
                        help='number of frames to share with the previous image')
    parser.add_argument('--annotation_path', type=str,
                        help='path of the annotation files')
    ap = argparse.ArgumentParser()
    args = vars(ap.parse_args())

    skeleton_path = 'videos'
    max_frames = 68
    stride = 20
    # skeleton_path = args["skeleton_path"]
    # max_frames = args["max_frames"]
    # stride = args["stride"]
    generate_images(skeleton_path, max_frames, stride)

if __name__ == "__main__":
    main()