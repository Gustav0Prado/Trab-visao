#!/usr/bin/python3

import cv2, sys, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

WINDOW_SIZE   = 4
NUM_CLUSTERS  = 3
RANDOM_NUMBER = random.randint(0, 100)

# Filtros baseados em derivadas de tamanho 3, 5 e 7
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

#################################################################################################

def apply_filters(img):
    kernel_results = {'Original: ': img}
    for kernel_name, kernel in filtros.items():
        filtered = cv2.filter2D(img, cv2.CV_64F, kernel)
        
        filtered = np.abs(filtered)
        filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
        filtered = filtered.astype(np.uint8)
        
        kernel_results[kernel_name] = filtered
    return kernel_results

#################################################################################################

def sliding_window(kernel_results, img):
    height, width = img.shape
    features = []

    for y in range(0, height, WINDOW_SIZE):
        for x in range(0, width, WINDOW_SIZE):
            vet = []
            for filter,res in kernel_results.items():
                sliding_window = res[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                mean = np.mean(sliding_window)
                vet.append(mean)   
            features.append(vet)
            
    return features
            
#################################################################################################            

def k_means(features):
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_NUMBER)
    labels = kmeans.fit_predict(features)
    
    return labels

#################################################################################################

def plot_image(labels, img):
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
    alpha = 0.5  # transparência

    height, width = img.shape
    i = 0
    for y in range(0, height, WINDOW_SIZE):
        for x in range(0, width, WINDOW_SIZE):
            cor = cores[labels[i] % len(cores)]
            overlay[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] = cor
            i += 1

    # Combinar overlay com a imagem original
    img_segmentada = cv2.addWeighted(overlay, alpha, img_colorida, 1 - alpha, 0)
    kernel_results['k-means'] = img_segmentada

    # Plotar a imagem com segmentação
    plt.figure(figsize=(16, 9))  # Tamanho da figura
    
    # Plota todos os filtros
    # for i, (name, res) in enumerate(kernel_results.items()):
    #     plt.subplot(5, 5, i+1)
    #     plt.imshow(res, cmap='gray')
    #     plt.title(name)
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    
    plt.imshow(img_segmentada, cmap='gray')
    plt.title("Imagem segmentada com K-Means")
    plt.axis('off')
    plt.show()
    

###########################################################################################
# Função principal

image_path = sys.argv[1]

# Processa imagem
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
kernel_results = apply_filters(img)
features = sliding_window(kernel_results, img)
labels = k_means(features)
plot_image(labels, img)