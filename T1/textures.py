#!/usr/bin/python3

import cv2, sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

image_path = sys.argv[1]
image_list = glob(image_path)

num_scales = 3

# Filtros baseados em derivadas
# Não sei se estão certos
filtros = {
    'Derivada Horizontal': np.array([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]),

    'Derivada Vertical': np.array([[ 1,  2,  1],
                                   [ 0,  0,  0],
                                   [-1, -2, -1]]),

    'Derivada 45°': np.array([[ 0,  1,  2],
                              [-1,  0,  1],
                              [-2, -1,  0]]),

    'Derivada 135°': np.array([[ 2,  1,  0],
                               [ 1,  0, -1],
                               [ 0, -1, -2]]),

    'Circular': np.array([[ 0, -1,  0],
                          [-1,  4, -1],
                          [ 0, -1,  0]])
}

# Processa e exibe as imagens
for caminho in image_list:
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

    # Armazena todos os resultados
    all_results = []

    # Cria cópia da imagem original
    current_img = img.copy()

    # Processa a imagem em num_scales escalas
    for scale in range(num_scales):
        kernel_results = {}
        for kernel_name, kernel in filtros.items():
            filtered = cv2.filter2D(current_img, -1, kernel)
            kernel_results[kernel_name] = filtered
        
        all_results.append((scale, current_img.shape, kernel_results))

        # Aplica suavização Gaussiana
        blurred = cv2.GaussianBlur(current_img, (5, 5), sigmaX=1)

        # Reduz a imagem para próxima escala
        current_img = cv2.pyrDown(blurred)

    # Mostra todas as imagens no mesmo plot
    fig, axes = plt.subplots(num_scales, 5, figsize=(20, 9))
    for scale, shape, results in all_results:
        for kernel_name, img in results.items():
            row = scale
            col = list(filtros.keys()).index(kernel_name)
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Escala {scale} - {kernel_name}')
            ax.axis('off')
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.show()