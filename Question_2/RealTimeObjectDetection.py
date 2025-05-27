import cv2
import numpy as np
import time

net = cv2.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1]
                 for i in net.getUnconnectedOutLayers().flatten()]


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]


    blob = cv2.dnn.blobFromImage(frame,
                                 scalefactor=1/255.0,
                                 size=(500, 500),
                                 swapRB=True,
                                 crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []


    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf > 0.3:
                cx, cy, bw, bh = (det[0:4] * np.array([w, h, w, h])).astype(int)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(conf)
                class_ids.append(class_id)


    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.35)


    if len(idxs) > 0:
        for idx in idxs.flatten():
            x, y, bw, bh = boxes[idx]
            label = f"{classes[class_ids[idx]]}: {confidences[idx]:.2f}"
            cv2.rectangle(frame,
                          (x, y),
                          (x + bw, y + bh),
                          (0, 255, 0), 2)
            cv2.putText(frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0), 2)


    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255), 2)


    cv2.imshow("YOLOv3-Tiny Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
