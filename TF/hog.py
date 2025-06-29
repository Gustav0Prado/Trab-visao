#!/usr/bin/python3

import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(sys.argv[1])

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

bboxes, weights = hog.detectMultiScale(
    img,
    winStride=(4, 4),      # Passos menores ajudam, mas são mais lentos
    padding=(8, 8),        # Padding em volta da janela
    scale=1.05             # Escala entre pirâmides de imagem
)

res = img.copy()

for i, (x, y, w, h) in enumerate(bboxes):
    if weights[i] > 0.6:  # ajuste o limiar conforme necessário
        cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(f"Número de pessoas: {len(bboxes)}")

img_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title(f"Pessoas detectadas: {len(bboxes)}")
plt.axis("off")
plt.show()