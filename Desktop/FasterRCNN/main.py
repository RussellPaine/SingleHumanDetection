import tensorflow as tf
import cv2
import os
import numpy


# def get_graph(graph_file):
#     with tf.gfile.FastGFile(graph_file, "rb") as file:
#         graph= tf.GraphDef()
#         graph.ParseFromString(file.read())
#     return graph

def random_colors(N):
    numpy.random.seed(1)
    colors = [tuple(255 * numpy.random.rand(3)) for _ in range(N)]
    return colors


def display_instances(image, boxes, masks, ids, names, scores):

    n_instances = boxes.shape[0]

    if not n_instances:
        print('Nothing Here')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    
    colors = random_colors(n_instances)

    height, width = image.shape[:2]

    for i, color in enumerate(colors):
        if not numpy.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]

        image = numpy.array(image)

        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, label, (x1,y1), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    return image


if  __name__ == "__main__":


    modelDir = os.path.join(os.getcwd(), "model", "saved_model")
    model = tf.compat.v2.saved_model.load(modelDir)

    # pbFilePath = os.path.join(os.getcwd(), "model", "frozen_inference_graph.pb")
    # graph = get_graph(pbFilePath)

    # config = tf.ConfigProto()
    # session = tf.Session(config=config)
    # tf.import_graph_def(graph, name='')


    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = capture.read()
        scores, boxes, classes, num_detections = model.run()


    capture.release()
    cv2.destroyAllWindows()