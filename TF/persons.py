#!/usr/bin/python3

from ultralytics import YOLO
import matplotlib.pyplot as plt
import sys, cv2

# Carrega o modelo YOLO
model = YOLO('yolov11l-face.pt')

# Roda a detecção
results = model(sys.argv[1])
img = cv2.imread(sys.argv[1])

# Pega as detecções
detections = results[0].boxes
classes = detections.cls.cpu().numpy()  # índices das classes detectadas

# Conta quantas vezes a classe 0 (pessoa) apareceu
num_pessoas = sum(cls == 0 for cls in classes)
print(f'Pessoas detectadas: {num_pessoas}')

# Pega as caixas do primeiro resultado
boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # coordenadas (x1, y1, x2, y2)

# Desenha apenas as caixas
for box in boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# Converte de BGR para RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Exibe com matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.axis("off")
plt.title(f"Pessoas detectadas: {num_pessoas}")
plt.show()