#!/usr/bin/python3

import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(sys.argv[1])
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img)
points = plt.ginput(4)
plt.close()

# Pega todos os pontos indicados pelo usuário
pt_A = points[0]
pt_B = points[1]
pt_C = points[2]
pt_D = points[3]

# Calcula as distâncias entre os pontos 2 a 2 e pega a menor
width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))

# Menor distância será usada como largura da imagem final
minWidth = min(int(width_AD), int(width_BC))
 
# Calcula as distâncias entre os pontos 2 a 2 e pega a menor
height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))

# Menor distância será usada como altura da imagem final
minHeight = min(int(height_AB), int(height_CD))

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, minHeight - 1],
                        [minWidth - 1, minHeight - 1],
                        [minWidth - 1, 0]])


# Cria matriz de transformação M
M = cv2.getPerspectiveTransform(input_pts,output_pts)

# Realiza mudança de perspectiva
out = cv2.warpPerspective(img,M,(minWidth, minHeight))

# Plota resultado com perspectiva corrigida
plt.imshow(out)
plt.show()