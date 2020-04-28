import tensorflow as tf
import cv2
import os
import numpy as np
import sys
from imutils.video import VideoStream
from imutils.video import FPS
import time
from openvino.inference_engine import IENetwork, IECore, IEPlugin
from PIL import Image

CWD_PATH = os.getcwd()
MODEL_xml = os.path.join(os.getcwd(), "model", "frozen_inference_graph.xml")
MODEL_bin = os.path.join(os.getcwd(), "model", "frozen_inference_graph.bin")
labelsPath = os.path.join(os.getcwd(), "model", "classes.txt")
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

if  __name__ == "__main__":

	# OpenVinoIE = IECore()
	# print("Available Devices: ", OpenVinoIE.available_devices)

	# net = IENetwork(model=MODEL_xml, weights=MODEL_bin)
	# net.add_outputs("detection_output")
	# input_blob = next(iter(net.inputs))
	# out_blob = next(iter(net.outputs))
	
	# plugin = IEPlugin(device="MYRIAD")
	# exec_net = plugin.load(network=net)
	# H, W = net.inputs[input_blob].shape



	OpenVinoIE = IECore()
	print("Available Devices: ", OpenVinoIE.available_devices)

	net = IENetwork(model=MODEL_xml, weights=MODEL_bin)

	feed_dict = {}
	for blob_name in net.inputs:
		if len(net.inputs[blob_name].shape) == 4:
			input_blob = blob_name
		elif len(net.inputs[blob_name].shape) == 2:
			img_info_input_blob = blob_name

	out_blob = next(iter(net.outputs))
	exec_net = OpenVinoIE.load_network(network=net, device_name="MYRIAD")

	n, c, h, w = net.inputs[input_blob].shape

	print(n, c, h, w)

	frame_rate_calc = 1
	freq = cv2.getTickFrequency()

	camera = VideoStream(usePiCamera=True).start()
	time.sleep(2.0)
	fps = FPS().start()

	while True:
		t1 = cv2.getTickCount()

		frame = camera.read()

		in_frame = cv2.resize(frame, (w, h))
		in_frame = in_frame.transpose((2, 0, 1))
		in_frame = in_frame.reshape((n, c, h, w))
		

		results = exec_net.infer(inputs={input_blob: in_frame})

		cv2.imshow("frame", frame)

		t2 = cv2.getTickCount()
		time1 = (t2-t1)/freq
		frame_rate_calc = 1/time1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
cv2.destroyAllWindows()

