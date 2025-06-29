#!/usr/bin/python3

import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(sys.argv[1])
(h, w) = img.shape[:2]

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

res = img.copy()
count = 0

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.12:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        cv2.rectangle(res, (x1, y1), (x2, y2), (0, 255, 0), 2)
        count += 1

img_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title(f"Pessoas detectadas: {count}")
plt.axis("off")
plt.show()