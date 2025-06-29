#!/usr/bin/python3

import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(sys.argv[1])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

res = img.copy()

for (x, y, w, h) in faces:
    cv2.rectangle(res, (x, y), (x+w, y+h), (0, 255, 0), 2)

print(f"Número de pessoas: {len(faces)}")

img_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title(f"Pessoas detectadas: {len(faces)}")
plt.axis("off")
plt.show()