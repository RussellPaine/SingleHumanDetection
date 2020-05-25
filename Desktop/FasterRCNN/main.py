import tensorflow as tf
import cv2
import os
import numpy as np
import sys
from filterpy.kalman import KalmanFilter


np.random.seed(0)
MODEL_NAME = 'model/faster_rcnn_inception_v2_coco_2018_01_28'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
labelsPath = os.path.join(os.getcwd(), "model", "classes.txt")
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


def convert_bbox_to_xyYa(bbox):
	width = bbox[2] - bbox[0]
	height = bbox[3] - bbox[1]
	x = bbox[0] + width / 2.  # midX
	y = bbox[1] + height / 2.  # midY
	a = width * height  # Area
	Y = width / float(height)  # Aspect Ratio
	return np.array([x, y, a, Y]).reshape((4, 1))


def convert_xyYh_to_bbox(xyYa, score=None):
	width = np.sqrt(xyYa[2] * xyYa[3])
	heigth = xyYa[2] / width
	x1 = xyYa[0] - width / 2.
	y1 = xyYa[1] - heigth / 2.
	x2 = xyYa[0] + width / 2.
	y2 = xyYa[1] + heigth / 2.
	if(score == None):
		return np.array([x1, y1, x2, y2]).reshape((1, 4))
	else:
		return np.array([x1, y1, x2, y2, score]).reshape((1, 5))

def get_bboxes(frame, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_expanded = np.expand_dims(frame_rgb, axis=0)

	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: frame_expanded})

	boxes = np.squeeze(boxes)
	classes = np.squeeze(classes).astype(np.int32)
	scores = np.squeeze(scores)

	height, width = frame.shape[:2]
	bboxes = []
	for i in range(boxes.shape[0]):
		if classes[i] != 1:
			continue
		if scores[i] > 0.85:
			y1, x1, y2, x2 = boxes[i] * np.array([height, width, height, width])
			bboxes.append([x1, y1, x2, y2, scores[i]])
	return np.array(bboxes)

def match_Predictions(detections, trackers, iou_threshold=0.3):
	if(len(trackers) == 0):
		return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
	iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

	for d,det in enumerate(detections):
		for t,trk in enumerate(trackers):
			iou_matrix[d,t] = iou(det,trk)

	if min(iou_matrix.shape) > 0:
		a = (iou_matrix > iou_threshold).astype(np.int32)
		if a.sum(1).max() == 1 and a.sum(0).max() == 1:
			matched_indices = np.stack(np.where(a), axis=1)
		else:
			matched_indices = linear_assignment(-iou_matrix)
	else:
		matched_indices = np.empty(shape=(0,2))

	unmatched_detections = []
	for d, det in enumerate(detections):
		if(d not in matched_indices[:,0]):
			unmatched_detections.append(d)
	unmatched_trackers = []

	for t, trk in enumerate(trackers):
		if(t not in matched_indices[:,1]):
			unmatched_trackers.append(t)

	matches = []
	for m in matched_indices:
		if(iou_matrix[m[0], m[1]]<iou_threshold):
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else:
			matches.append(m.reshape(1,2))

	if(len(matches)==0):
		matches = np.empty((0,2),dtype=int)
	else:
		matches = np.concatenate(matches,axis=0)

	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def linear_assignment(cost_matrix):
	from scipy.optimize import linear_sum_assignment
	x, y = linear_sum_assignment(cost_matrix)
	return np.array(list(zip(x, y)))

def iou(bb_test, bb_gt):
	xx1 = np.maximum(bb_test[0], bb_gt[0])
	yy1 = np.maximum(bb_test[1], bb_gt[1])
	xx2 = np.minimum(bb_test[2], bb_gt[2])
	yy2 = np.minimum(bb_test[3], bb_gt[3])
	w = np.maximum(0., xx2 - xx1)
	h = np.maximum(0., yy2 - yy1)
	wh = w * h
	o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
	return(o)

class MOTTracker(object):
	def __init__(self):
		self.death = 1
		self.uncertainty = 3
		self.trackers = []
		self.frameCount = 0

	def update(self, bboxes=np.empty((0, 5))):
		self.frameCount += 1
		predictedTracks = np.zeros((len(self.trackers), 5))

		deaths = []
		tracked = []

		
		for t, tracking in enumerate(predictedTracks):
			potentialPos = self.trackers[t].predict()[0]
			tracking[:] = [potentialPos[0], potentialPos[1], potentialPos[2], potentialPos[3], 0]
			if np.any(np.isnan(potentialPos)):
				deaths.append(t)

		predictedTracks = np.ma.compress_rows(np.ma.masked_invalid(predictedTracks))
		for t in reversed(deaths):
			self.trackers.pop(t)

		matched, unmatched_dets, unmatched_trks = match_Predictions(bboxes, predictedTracks)

		for m in matched:
			self.trackers[m[1]].update(bboxes[m[0], :])

		for i in unmatched_dets:
			trk = kalmanTracker(bboxes[i,:])
			self.trackers.append(trk)

		i = len(self.trackers)

		for trk in reversed(self.trackers):
			d = trk.get_state()[0]
			h = trk.get_history()

			if (trk.time_since_update < 1) and (trk.hit_streak >= self.uncertainty or self.frameCount <= self.uncertainty):
				tracked.append(person(d, trk.id + 1, h)) 
			i -= 1
			if(trk.time_since_update > self.death):
				self.trackers.pop(i)

		if(len(tracked) > 0):
			return tracked
		tracked.append(person([], -1, []))
		return tracked

class kalmanTracker(object):
	ids = 0
	def __init__(self, bbox):
		self.KF = KalmanFilter(dim_x= 7, dim_z= 4)
		self.KF.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
		self.KF.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

		self.KF.R[2:,2:] *= 10.
		self.KF.P[4:,4:] *= 1000.
		self.KF.P *= 10.
		self.KF.Q[-1,-1] *= 0.01
		self.KF.Q[4:,4:] *= 0.01

		self.KF.x[:4] = convert_bbox_to_xyYa(bbox)

		self.time_since_update = 0
		self.id = kalmanTracker.ids
		kalmanTracker.ids += 1

		self.history = []
		self.trackhistory = []
		self.hits = 0
		self.hit_streak = 0
		self.age = 0

	def update(self, bbox):
		self.time_since_update = 0
		for hist in self.history:
			self.trackhistory.append(hist)
		self.history = []
		self.hits += 1
		self.hit_streak += 1
		self.KF.update(convert_bbox_to_xyYa(bbox))

	def predict(self):
		if((self.KF.x[6] + self.KF.x[2]) <= 0):
			self.KF.x[6] *= 0.0
		self.KF.predict()
		self.age += 1
		if(self.time_since_update > 0):
			self.hit_streak = 0
		self.time_since_update += 1
		self.history.append(convert_xyYh_to_bbox(self.KF.x))
		return self.history[-1]

	def get_state(self):
		return convert_xyYh_to_bbox(self.KF.x)

	def get_history(self):
		h = []
		for hist in self.trackhistory:
			h.append(hist[0])
		return h

def drawPersons(frame, trackers):
	for person in trackers:
		if person.classID != -1:
			color = COLORS[person.classID]
			color = [int(c) for c in color]
			cv2.putText(frame, str(person.classID), (int(person.bbox[0]) + 5, int(person.bbox[1]) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.rectangle(frame, (int(person.bbox[0]), int(person.bbox[1])), (int(person.bbox[2]), int(person.bbox[3])), color, 2)
			prevX, prevY = -1, -1
			for h in person.history:
				x = h[0] + (h[2] - h[0]) / 2.
				y = h[1] + (h[3] - h[1]) / 2.
				cv2.circle(frame, (int(x), int(y)), 3, color, 2)
				if prevX != -1 and prevY != -1:
					cv2.line(frame, (prevX, prevY), (int(x), int(y)), color, 2)
				prevX = int(x)
				prevY = int(y)
	return frame

class person():
	def __init__(self, bbox, classID, history):
		self.bbox = bbox
		self.classID = classID
		self.history = history

if  __name__ == "__main__":
	print("Num Of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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

tracker = MOTTracker()

frame_rate_calc = 1
freq = cv2.getTickFrequency()

camera = cv2.VideoCapture(0)

while(True):

	t1 = cv2.getTickCount()
	ret, frame = camera.read()

	bboxes = get_bboxes(frame, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)	
	trackers = tracker.update(bboxes)

	frame = drawPersons(frame, trackers)

	# for d in trackers:
	# 	print('%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))
	# 	cv2.putText(frame, str(d[4]), (int(d[0]), int(d[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

	cv2.putText(frame, "{0:.2f}".format(frame_rate_calc), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
	cv2.imshow('frame', frame)
	t2 = cv2.getTickCount()
	time1 = (t2-t1)/freq
	frame_rate_calc = 1/time1

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

camera.release()
cv2.destroyAllWindows()
