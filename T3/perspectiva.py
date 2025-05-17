import cv2, sys, random
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("corredor.jpeg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img)
points = plt.ginput(4)
plt.close()

# All points are in format [cols, rows]
pt_A = points[0]
pt_B = points[1]
pt_C = points[2]
pt_D = points[3]

# Here, I have used L2 norm. You can use L1 also.
width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))
 
height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])


# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts,output_pts)

out = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

#plot
plt.imshow(out)
plt.show()