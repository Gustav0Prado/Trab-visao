import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


filtros = {
    "horizontal": np.array([
        [-1, -1, -1, -1, -1],
        [-2, -2, -2, -2, -2],
        [ 0,  0,  0,  0,  0],
        [ 2,  2,  2,  2,  2],
        [ 1,  1,  1,  1,  1]
    ]),
    "vertical": np.array([
        [-1, -2,  0,  2,  1],
        [-1, -2,  0,  2,  1],
        [-1, -2,  0,  2,  1],
        [-1, -2,  0,  2,  1],
        [-1, -2,  0,  2,  1]
    ]),
    "45 graus": np.array([
        [ 0,  0, -1, -2,  0],
        [ 0, -1, -2,  0,  2],
        [-1, -2,  0,  2,  1],
        [-2,  0,  2,  1,  0],
        [ 0,  2,  1,  0,  0]
    ]),
    "135 graus": np.array([
        [ 0,  2,  1,  0,  0],
        [-2,  0,  2,  1,  0],
        [-1, -2,  0,  2,  1],
        [ 0, -1, -2,  0,  2],
        [ 0,  0, -1, -2,  0]
    ]),
    "circular": np.array([
        [-1, -1, -1, -1, -1],
        [-1,  1,  2,  1, -1],
        [-1,  2,  4,  2, -1],
        [-1,  1,  2,  1, -1],
        [-1, -1, -1, -1, -1]
    ])
}

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
resultados = {'original': img}

# aplicando filtros
for kernel_name, kernel in filtros.items():
    resultados[kernel_name] = cv2.filter2D(img, -1, kernel)

# calculando vetores das janelas
altura, largura = img.shape
tam_kernel = 5
vetores = []
for y in range(0, altura, tam_kernel):
    for x in range(0, largura, tam_kernel):
        vet = []
        for filtro,res in resultados.items():
            janela = res[y:y+tam_kernel, x:x+tam_kernel]
            media = np.mean(janela)
            vet.append(media)   
        vetores.append(vet)

print(len(vetores))

# k-means
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(vetores)

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

i = 0
for y in range(0, altura, tam_kernel):
    for x in range(0, largura, tam_kernel):
        cor = cores[labels[i] % len(cores)]
        overlay[y:y+tam_kernel, x:x+tam_kernel] = cor
        i += 1

# Combinar overlay com a imagem original
img_segmentada = cv2.addWeighted(overlay, alpha, img_colorida, 1 - alpha, 0)
resultados['k-means'] = img_segmentada

# plotando imagens
plt.figure(figsize=(16, 9))  # Tamanho da figura

for i, (nome,img) in enumerate(resultados.items()):
    # plt.subplot(3, 3, i+1)  # 3 linhas, 3 colunas
    plt.imshow(img, cmap='gray')
    plt.title(nome)
    plt.axis('off')

plt.tight_layout()
plt.show()