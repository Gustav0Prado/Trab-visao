#!/usr/bin/python3

import cv2, sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

caminho_imagens = sys.argv[1]
lista_imagens = glob(caminho_imagens)

# Filtros baseados em derivadas
# Fazer em escalas diferentes!!
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

    # Aplica cada kernel
    for nome, kernel in filtros.items():
        if kernel is not None:
            resposta = cv2.filter2D(img, -1, kernel)
            respostas[nome] = resposta

    # Exibe as imagens em 2 linhas com imagens maiores
    total = len(respostas)
    colunas = 3
    linhas = (total + 1) // 2

    plt.figure(figsize=(20,10))
    for i, (nome, imagem) in enumerate(respostas.items()):
        plt.subplot(linhas, colunas, i + 1)
        plt.imshow(imagem, cmap='gray')
        plt.title(nome, fontsize=14)
        plt.axis('off')
    plt.show()