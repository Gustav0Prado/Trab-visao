#!/usr/bin/python3

import cv2, sys, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Filtros baseados em derivadas
filtros = {
    'Derivada Horizontal': np.array([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]),
    
    'Derivada Horizontal 2': np.array([[-1, -2,  0,  2,  1],
                                       [-4, -8,  0,  8,  4],
                                       [-6, -12, 0, 12, 6],
                                       [-4, -8,  0,  8,  4],
                                       [-1, -2,  0,  2,  1]]),
    
    'Derivada Horizontal 3': np.array([[-1, -4, -5,  0,  5,  4,  1],
                                       [-6, -24, -30, 0, 30, 24, 6],
                                       [-15, -60, -75, 0, 75, 60, 15],
                                       [-20, -80, -100, 0, 100, 80, 20],
                                       [-15, -60, -75, 0, 75, 60, 15],
                                       [-6, -24, -30, 0, 30, 24, 6],
                                       [-1, -4, -5,  0,  5,  4,  1]]),

    'Derivada Vertical': np.array([[ 1,  2,  1],
                                   [ 0,  0,  0],
                                   [-1, -2, -1]]),
    
    'Derivada Vertical 2': np.array([[-1, -4, -6, -4, -1],
                                     [-2, -8, -12, -8, -2],
                                     [ 0,  0,   0,  0,  0],
                                     [ 2,  8,  12,  8,  2],
                                     [ 1,  4,   6,  4,  1]]),
    
    'Derivada Vertical 3': np.array([[-1, -6, -15, -20, -15, -6, -1],
                                     [-4, -24, -60, -80, -60, -24, -4],
                                     [-5, -30, -75, -100, -75, -30, -5],
                                     [ 0,   0,   0,    0,   0,   0,  0],
                                     [ 5,  30,  75,  100,  75,  30,  5],
                                     [ 4,  24,  60,   80,  60,  24,  4],
                                     [ 1,   6,  15,   20,  15,   6,  1]]),

    'Derivada 45° 1': np.array([[-2, -1,  0],
                                [-1,  0,  1],
                                [ 0,  1,  2]]),

    'Derivada 45° 2' : np.array([[-2, -6,  0,  6,  2],
                                [-6, -24, 0, 24, 6],
                                [ 0,  0,   0,  0,  0],
                                [ 6, 24,  0, -24, -6],
                                [ 2,  6,  0, -6,  -2]]),
    
    'Derivada 45° 3' : np.array([[-2,  -8,  -6,   0,   6,   8,  2],
                                [-6, -32, -24,   0,  24,  32,  6],
                                [-12, -48, -36,  0,  36,  48, 12],
                                [ 0,   0,   0,   0,   0,   0,  0],
                                [ 12,  48,  36,  0, -36, -48, -12],
                                [ 6,  32,  24,   0, -24, -32, -6],
                                [ 2,   8,   6,   0,  -6,  -8, -2]]),

    'Derivada 135° 1': np.array([[ 0,  1,  2],
                                 [-1,  0,  1],
                                 [-2, -1,  0]]),
    
    'Derivada 135° 2': np.array([[ 2,  6,  0, -6, -2],
                                 [ 6, 24,  0, -24, -6],
                                 [ 0,  0,  0,   0,  0],
                                 [-6, -24, 0,  24,  6],
                                 [-2, -6,  0,   6,  2]]),
    
    'Derivada 135° 3': np.array([[ 2,   8,   6,  0,  -6,  -8,  -2],
                                 [ 6,  32,  24, 0, -24, -32,  -6],
                                 [12,  48,  36, 0, -36, -48, -12],
                                 [ 0,   0,   0, 0,   0,   0,   0],
                                 [-12, -48, -36, 0, 36, 48, 12],
                                 [-6, -32, -24, 0, 24, 32, 6],
                                 [-2,  -8,  -6, 0,  6, 8, 2]]),

    'Circular 1': np.array([[ 0,-1, 0],
                            [-1, 4, -1],
                            [ 0,-1, 0]]),
    
    'Circular 2': np.array([[ 0,  0, -1,  0,  0],
                            [ 0,  0, -1,  0,  0],
                            [-1, -1,  8, -1, -1],
                            [ 0,  0, -1,  0,  0],
                            [ 0,  0, -1,  0,  0]]),
    
    'Circular 3': np.array([[ 0,  0,  0, -1,  0,  0,  0],
                            [ 0,  0,  0, -1,  0,  0,  0],
                            [ 0,  0,  0, -1,  0,  0,  0],
                            [-1, -1, -1, 12, -1, -1, -1],
                            [ 0,  0,  0, -1,  0,  0,  0],
                            [ 0,  0,  0, -1,  0,  0,  0],
                            [ 0,  0,  0, -1,  0,  0,  0]]),
}

def plot_images(kernel_results):
    # Plotar as imagens de saída
    # Define número de colunas desejado
    cols = 5
    rows = math.ceil(len(filtros) / cols)

    # Cria os subplots (grid de rows x cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.reshape(rows, cols).T.flatten()

    # Preenche os plots
    for i, (nome, resultado) in enumerate(kernel_results.items()):
        axes[i].imshow(resultado, cmap='gray')
        axes[i].set_title(nome)
        axes[i].axis('off')

    # Desliga os plots extras (caso sobrem espaços vazios)
    for j in range(len(kernel_results), len(axes)):
        axes[j].axis('off')

    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.show()
    
def normalize_image(filtered):
    # Remove valores negativos
    filtered = np.abs(filtered)
    
    # Normaliza imagem
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
    filtered = filtered.astype(np.uint8)
    
    return filtered

def apply_filters(img):
    kernel_results = {}
    for kernel_name, kernel in filtros.items():
        filtered = cv2.filter2D(img, cv2.CV_64F, kernel)
        # filtered = normalize_image(filtered)
        
        # Insere no dic de resultados
        kernel_results[kernel_name] = filtered
    return kernel_results

def sliding_window(kernel_results, img):
    height, width = img.shape
    features = []

    window_size = 5

    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            vet = []
            for filter,res in kernel_results.items():
                sliding_window = res[y:y+window_size, x:x+window_size]
                mean = np.mean(sliding_window)
                vet.append(mean)   
            features.append(vet)
            
    return features
            
            
def k_means(n_clusters, features, img):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(features)

    # Reconstrução da imagem segmentada
    cores = [ # Cores para os clusters
        (255, 0, 0),    # vermelho
        (0, 255, 0),    # verde
        (0, 0, 255),    # azul
        (255, 255, 0),  # amarelo
        (0, 255, 255),  # ciano
        (255, 0, 255),  # magenta
    ]

    # Criar imagem de overlay colorido
    img_colorida = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = img_colorida.copy()
    alpha = 0.3  # transparência
    
    height, width = img.shape

    window_size = 5

    i = 0
    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            cor = cores[labels[i] % len(cores)]
            overlay[y:y+window_size, x:x+window_size] = cor
            i += 1

    # Combinar overlay com a imagem original
    img_segmentada = cv2.addWeighted(overlay, alpha, img_colorida, 1 - alpha, 0)
    kernel_results['k-means'] = img_segmentada

    # plotando imagens
    plt.figure(figsize=(16, 9))  # Tamanho da figura

    for i, (nome,img) in enumerate(kernel_results.items()):
        # plt.subplot(3, 3, i+1)  # 3 linhas, 3 colunas
        plt.imshow(img, cmap='gray')
        plt.title(nome)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

###########################################################################################
# Função principal

image_path  = sys.argv[1]

# Processa e exibe as imagens
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

kernel_results = apply_filters(img)

features = sliding_window(kernel_results, img)

# k-means
k_means(2, features, img)