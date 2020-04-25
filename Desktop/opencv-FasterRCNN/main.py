import numpy as np
import cv2
import os

labelsPath = os.path.join(os.getcwd(), "model", "classes.txt")
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.join(os.getcwd(), "model", "frozen_inference_graph.pb")
configPath = os.path.join(os.getcwd(), "model", "config.pbtxt")

print("[INFO] loading Faster R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

capture = cv2.VideoCapture(0)
writer = None

while True:
    ret, frame = capture.read()
    if not ret:
        continue

    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes) = net.forward(["detection_out_final"])

    for i in range(0, boxes[0].shape[2]):
        classID = int(boxes[0][0, 0, i, 1])
        if classID != 0:
            continue
        confidence = boxes[0][0, 0, i, 2]
        if confidence > 0.9:
            (H, W) = frame.shape[:2]
            box = boxes[0][0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            roi = frame[startY:endY, startX:endX]
            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
            frame[startY:endY, startX:endX]= blended
            color = [int(c) for c in color]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break