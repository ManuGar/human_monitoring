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





# Para generar las imágenes necesitamos el path de la carpeta en la que estan los frames de los esqueletos de los videos,
# los frames que vamos a guardar en cada imagen y la cantidad de frames que va a compartir con la imagen siguiente
def generate_images(skeleton_path, annotation_path, max_frames=68, stride=20):
    # Límite de frames que se van a incluir en cada imagen
    # max_frames = 128 #Esto es por el resultado que nos ha dado la media y la mediana
    count_generated_frames=0

    # En este vector guardamos el orden de los joint para luego poder recorrerlo en el orden que queremos
    joints_order = [1,2,3,26,27,28,29,28,27,30,31,30,27,26,3,2,4,5,6,7,8,9,8,7,10,7,6,5,4,2,11,12,13,14,
                    15,16,15,14,17,14,13,12,11,2,1]
    # Plantilla de la imagen vacía (llena de 0) para ir completándola en cada paso
    img = np.zeros((max_frames, len(joints_order),3), dtype=np.uint8)
    annotation_file = ""
    contador_frames_video = 0
    i = 0 # Contador para el número de frames tratados. Al llegar al tope se vuelca la imagen generada a disco
    l = 0 # Contador para recorrer los frames de un mismo vídeo con un while y poder hacer bien el stride

    # Lista con todos los frames de los vídeos sobre los que se va a trabajar. Luego hay que tratar las rutas para saber
    # cuando se cambia de vídeo y demás.
    skeleton_path = list(paths.list_files(skeleton_path, validExts=".txt"))
    # Esta variable sirve para comprobar cuando estamos cambiando de video y reiniciar ciertas variables
    cambio_vid = ""

    # Recorremos la lista de frames
    for fichero in skeleton_path:
        # Tratamos la ruta para obtener la carpeta de los videos y poder realizar diferentes cálculos
        dirs = fichero.split(os.path.sep)
        # Ruta con el nombre del vídeo con el que trabajamos
        nombre_directorio = os.path.join(dirs[0], dirs[1])

        # Aquí comprobamos que estamos en un vídeo nuevo
        if cambio_vid!= dirs[1]:
            cambio_vid=dirs[1]
            contador_frames_video = len(list(paths.list_files(fichero[:fichero.rfind(os.path.sep)], validExts=".txt")))

            id_video = nombre_directorio.split(os.path.sep)[1].split("_")[0]
            output_path = os.path.join(nombre_directorio, "annotation_images_labelSmoothing.csv")
            for ann_files in os.listdir(annotation_path):
                id_ann_file = ann_files.split("_")[0]
                if id_ann_file == id_video:
                    annotation_file = os.path.join(annotation_path, ann_files)
                    break

        # Vamos a generar las imágenes del video que trate mientras el contador sea menos al número de imágenes que
        # tiene ese vídeo y las imágenes generadas también sean menores (para casos como que estamos en el final del
        # vídeo o el vídeo es muy corto

        # while l < contador_frames_video and (count_generated_frames) < contador_frames_video:

        # Leemos el archivo del frame para trabajar con él
        frame = pd.read_csv(fichero, sep="\t", header=0)
        frame = pd.DataFrame(frame)
        joints={}

        # Aquí guardamos los joints y los datos que queremos en un diccionario para luego generar la imagen
        for j, line in frame.iterrows():
           # print(str(line[3]) + "\t" + str(line[4]) + "\t" + str(line[5]))
           # Los valores máximos y mínimos han sido seleccionados mirando los valores que obtienen los esqueletos
           # que ibamos a estudiar (espero que esos valores se conserven también en los otros vídeos)
           joints[int(line[1])]=[normalize(line[3],-200, 1000, 255, 0),normalize(line[4],-1000, 1000, 255, 0),
                                 normalize(line[5],1200, 2200, 255, 0)]

        # Guardamos el contenido de las componentes de los diferentes joints en las coordenadas de la imagen.
        # Para cada frame generamos una fila en la imagen
        for idx in range(len(joints_order)):
            # print(joints[joints_order[idx]][0])
            img[i-1,idx,0]= joints[joints_order[idx]][0]
            img[i-1,idx,1]= joints[joints_order[idx]][1]
            img[i-1,idx,2]= joints[joints_order[idx]][2]

        # Aquí vamos a escribir la matriz con los datos a la imagen para guardarla. Primero comprobamos que podamos
        # seguir guardando la matriz con los datos de la imagen que generamos. Si hemos llegado al límite de frames o
        # al final del fichero, lo que haremos es escribirla en la imagen.


        # Creamos la carpeta de las imágenes en el caso de que no exista
        # Guardamos las imágenes generadas en una carpeta para que esté organizada junto al vídeo que hacen
        # referencia y poder saber fácilmente la clase de cada fila de las imágenes.
        if not os.path.exists(os.path.join(nombre_directorio, "images")):
            os.mkdir(os.path.join(nombre_directorio, "images"))

        if i < max_frames :
            if(l >= contador_frames_video or i >= contador_frames_video):

                annotation_images.annotation_images_labelsmoothing(count_generated_frames,
                                            count_generated_frames + i, 12, annotation_file, output_path)
                # Guardamos el nombre que tendrá la imagen
                generated_image = os.path.join(nombre_directorio, "images",
                                               "from_" + str(count_generated_frames) + "_to_" + (
                                                   str(count_generated_frames + max_frames)))


                # count_generated_frames = count_generated_frames + max_frames - stride

                cv2.imwrite(generated_image + ".jpg", img)
                cv2.imwrite(generated_image + "X.jpg", img[:, :, 0])
                cv2.imwrite(generated_image + "Y.jpg", img[:, :, 1])
                cv2.imwrite(generated_image + "Z.jpg", img[:, :, 2])
                img = np.zeros((max_frames, len(joints_order), 3), dtype=np.uint8)
                l = 0
                count_generated_frames = 0
                i = 0

            i += 1
            l += 1
        else:
            # Guardamos el id del video a partir de la ruta, con el id podremos generar a la vez también las
            # anotaciones de las imágenes que se van generando
            # Con el id del vídeo que estamos tratando, buscamos la anotación del vídeo para poder generar la
            # anotación de las imágenes



            annotation_images.annotation_images_labelsmoothing(count_generated_frames,count_generated_frames+i,12,annotation_file,output_path)

            # Guardamos el archivo de la anotación de las imágenes en el mismo nivel que la carpeta de las imágenes
            # y la de los esqueletos
            # Aquí si cambiamos de problema o modificamos las clases que queremos estudiar habrá que cambiar ese 12
            # que hace referencia al número de clases del problema que estamos tratando
            # Con esta función generamos las anotaciones

            # Guardamos el nombre que tendrá la imagen
            generated_image = os.path.join(nombre_directorio, "images",
                                           "from_" + str(count_generated_frames) + "_to_" + (
                                               str(count_generated_frames + max_frames)))


            # Actualizamos el número de frames tratados
            count_generated_frames = count_generated_frames + max_frames - stride
            # Reiniciamos los contadores de los frames tratados, guardamos las imágenes y actualizamos el valor de l

            cv2.imwrite(generated_image + ".jpg", img)
            cv2.imwrite(generated_image + "X.jpg", img[:,:,0])
            cv2.imwrite(generated_image + "Y.jpg", img[:,:,1])
            cv2.imwrite(generated_image + "Z.jpg", img[:,:,2])
            img = np.zeros((max_frames, len(joints_order), 3), dtype=np.uint8)
            i = 1
            l = l - stride + 1


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
    annotation_path = "annot_renamed"
    # skeleton_path = args["skeleton_path"]
    # max_frames = args["max_frames"]
    # stride = args["stride"]
    generate_images(skeleton_path, annotation_path, max_frames, stride)

if __name__ == "__main__":
    main()