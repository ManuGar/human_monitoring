import argparse
import os

# Esta es la versión para label smoothing y así conservar la información sobre todas las clases. Guradamos en un CSV el
# % de cuanto pertenece a cada clase. De esta forma no se pierde nada de información para hacer el entrenamiento.
# Aunque sea un proceso más costoso
def annotation_images_labelsmoothing(min_frame, max_frame, number_classes, annotation_path, output_path):
    actions_count = {}
    for i in range(number_classes):
        actions_count[i+1] = 0

    ann_file = open(annotation_path,"r")


    for line in ann_file:
        line = line.split("\t")
        if(line[0]>=min_frame and line[0]<max_frame):
            actions_count[line[1]]+=1
        if(line[0]>max_frame):
            break

    number_frames = max_frame-min_frame
    if (os.path.exists(output_path)):
        output_path_csv = open(output_path, "a")
    else:
        output_path_csv = open(output_path, "w")
        head = "images"
        for i in range(number_classes):
            head+= ", class " + i
        output_path_csv.write(head)

    line_write = "from_"+min_frame+"_to_"+max_frame
    for cla in actions_count:
        line_write += ", "+str(actions_count[cla]/number_frames)
    output_path_csv.write(line_write)
    output_path_csv.close()



# En este caso, solo guardamos el porcentaje de la clase que más frames tiene en la imagen. Es un proceso más sencillo
# para realizar luego el entrenamiento.
def annotation_images(min_frame, max_frame, number_classes, annotation_path, output_path):
    actions_count = {}
    for i in range(number_classes):
        actions_count[i + 1] = 0

    ann_file = open(annotation_path,"r")

    for line in ann_file:
        line = line.split("\t")
        if(line[0]>=min_frame and line[0]<max_frame):
            actions_count[line[1]]+=1
        if(line[0]>max_frame):
            break

    id = 1
    max=0
    for i in actions_count:
        if actions_count[i]>max:
            max = actions_count[i]
            id = i

    if (os.path.exists(output_path)):
        output_path_csv = open(output_path, "a")
    else:
        output_path_csv = open(output_path, "w")
        head = "images"
        for i in range(number_classes):
            head += ", class " + i
        output_path_csv.write(head)

    line_write = "from_"+min_frame+"_to_"+max_frame+ ", "+id
    output_path_csv.write(line_write)
    output_path_csv.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_frame', type=int,
                        help='frame number at which the image starts')
    parser.add_argument('--max_frame', type=int,
                        help='frame number at which the image ends')
    parser.add_argument('--number_classes', type=int,
                        help='number of actions of the problem')
    parser.add_argument('--annotation_path', type=str,
                        help='path of the file with the annotation of the video')
    parser.add_argument('--output_path', type=str,
                        help='path of the output to store CSV with the annotation of the image')
    ap = argparse.ArgumentParser()
    args = vars(ap.parse_args())

    min_frame = args["min_frame"]
    max_frame = args["max_frame"]
    number_classes = args["number_classes"]
    annotation_path = args["annotation_path"]
    output_path = args["output_path"]

    annotation_images_labelsmoothing(min_frame, max_frame, number_classes, annotation_path, output_path)

if __name__ == "__main__":
    main()