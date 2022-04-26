import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(arg.parse_args())
    img = cv2.imread(args["image"])
    weights = "custom_weight/yolov3.weights"
    net = cv2.dnn.readNetFromDarknet("custom_weight/yolov3.cfg", weights)

    hight, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []; confidences = [];class_ids = []
    for output in output_layers:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .6, .4)
    for correct in indexes.flatten():
        x, y, w, h = boxes[correct]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
