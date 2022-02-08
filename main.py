import pandas as pd
import cv2

skeleton_path = ''
i = 0
frames = 30
count_generated_imgs=0
for ske in skeleton_path:
    file = pd.read_csv(ske, sep=" ", header=None)
    file = pd.DataFrame(file)
    joints=[]
    #     con esto tenemos que ir leyendo los esqueletos y crear la imagen con todos.
    #     Tendríamos que poner el límite de cuantos frames vamos a tener para generar cada imagen
    for j, line in file.iterrows():
        joints[line[1]]=[line[2],line[3],line[4]]
        # aquí tenemos que saber el orden en el que usar el arbol del esqueleto. Pero creo que la mejor forma de trabajar
        # es crear un diccionario que guarde como clave el id del joint y luego el resto de datos. Con esto es más facil
        # recorrerlos y luego poder generar la imagen. Para la imagen tenemos que saber como normalizar las distancias.
        # además de saber como crear la imagen, en el artículo lo ponen en escala de grises pero igual es mejor hacerlo
        # en rgb para conservar la información de cada componente.
    # Aquí creamos el recorrido del esqueleto y luego generamos la imagen

    if i< frames:
        i+=1
    else:
        generated_image="from_"+count_generated_imgs+"_to_"+ (count_generated_imgs+frames)
        count_generated_imgs+=frames
        i=0















