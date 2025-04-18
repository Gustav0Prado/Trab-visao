#!/usr/bin/python3

import cv2, sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

caminho_imagens = sys.argv[1]
lista_imagens = glob(caminho_imagens)

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
for caminho in lista_imagens:
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    respostas = {'Original': img}

    # Armazena todos os resultados
    all_results = []

    current_img = img.copy()

    for scale in range(num_scales):
        # Aplica suavização Gaussiana
        blurred = cv2.GaussianBlur(current_img, (5, 5), sigmaX=1)

        kernel_results = {}
        for name, kernel in filtros.items():
            filtered = cv2.filter2D(blurred, -1, kernel)
            kernel_results[name] = filtered
        
        all_results.append((scale, current_img.shape, kernel_results))

        # Reduz a imagem para próxima escala
        current_img = cv2.pyrDown(blurred)

    # Mostra todas as imagens no mesmo plot
    fig, axes = plt.subplots(num_scales, 5, figsize=(20, 9))
    for scale, shape, results in all_results:
        for name, img in results.items():
            row = scale
            col = list(filtros.keys()).index(name)
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Escala {scale} - {name}')
            ax.axis('off')
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.show()