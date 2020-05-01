import cv2
import os
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import time
from openvino.inference_engine import IENetwork, IECore, IEPlugin

CWD_PATH = os.getcwd()
MODEL_xml = os.path.join(os.getcwd(), "model", "frozen_inference_graph.xml")
MODEL_bin = os.path.join(os.getcwd(), "model", "frozen_inference_graph.bin")
labelsPath = os.path.join(os.getcwd(), "model", "classes.txt")
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

if  __name__ == "__main__":
    plugin = IEPlugin("MYRIAD")
    net = IENetwork(model=MODEL_xml, weights=MODEL_bin)
    plugin.set_config({"VPU_HW_STAGES_OPTIMIZATION": "YES"})    
        
        
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name

    out_blob = next(iter(net.outputs))
    print(out_blob)
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]
        
    print(net.inputs[input_blob].shape)
    exec_net = plugin.load(network=net)
    del net
        
        
        
    camera = cv2.VideoCapture(1)
    assert camera.isOpened(), "Can't open " + camera
    
    cur_request_id = 0
    
    while camera.isOpened():
        

        ret, frame = camera.read()
        if ret:
            frame_h, frame_w = frame.shape[:2]
        if not ret:
            break
        
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))
        feed_dict[input_blob] = in_frame

        start = time.time()
        results = exec_net.start_async(request_id=cur_request_id, inputs=feed_dict,)
        
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            end = time.time()
            inf_time = end - start
            print('Inference Time: {} Seconds Single Image'.format(inf_time))
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > 0.85:
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    class_id = int(obj[1])
                    # Draw box and label\class_id
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    det_label = LABELS[class_id] if LABELS else str(class_id)
                    cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        cv2.imshow("frame", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

