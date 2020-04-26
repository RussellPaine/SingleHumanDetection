import tensorflow as tf
import cv2
import os
import numpy as np
import sys

MODEL_NAME = 'model'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
labelsPath = os.path.join(os.getcwd(), "model", "classes.txt")
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


if  __name__ == "__main__":
	print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.compat.v1.GraphDef()
		with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

		sess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

camera = cv2.VideoCapture(0)
while(True):
	t1 = cv2.getTickCount()
	ret, frame = camera.read()

	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_expanded = np.expand_dims(frame_rgb, axis=0)

	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: frame_expanded})


	boxes = np.squeeze(boxes)
	classes = np.squeeze(classes).astype(np.int32)
	scores = np.squeeze(scores)


	for i in range(boxes.shape[0]):
		if scores[i] > 0.85:
			box = tuple(boxes[i].tolist())
			outString = ''
			className = LABELS[classes[i]]
			outString = str(className)
			outString = '{}: {}%'.format(outString, int(100*scores[i]))
			color= COLORS[classes[i]]



	cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
	cv2.imshow('frame', frame)

	t2 = cv2.getTickCount()
	time1 = (t2-t1)/freq
	frame_rate_calc = 1/time1

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

camera.release()
cv2.destroyAllWindows()