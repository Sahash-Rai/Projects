import cv2
import numpy as np
import os

# Load YOLO
if not os.path.exists('yolov3.weights') or not os.path.exists('yolov3.cfg') or not os.path.exists('classes.txt'):
    print("Error: Missing YOLO files. Ensure 'yolov3.weights', 'yolov3.cfg', and 'classes.txt' are in the working directory.")
    exit()

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load classes
classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

# Load video
cap = cv2.VideoCapture('test1.mp4')
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    ret, img = cap.read()
    if not ret or img is None:
        print("Error: Cannot read frame. Exiting...")
        break

    height, width, _ = img.shape

    # Preprocess frame for YOLO
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i % len(colors)]  # Prevent IndexError
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence}", (x, y + 20), font, 2, (255, 255, 255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
