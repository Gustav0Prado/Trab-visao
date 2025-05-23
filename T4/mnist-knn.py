#!/usr/bin/python3

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
                    prog='Programa de testes usando MNIST',
                    description='Realiza teste no dataset MNIST usando os parametros passados')

parser.add_argument('kneighbours')
parser.add_argument('metric')
parser.add_argument('test_size')

args = parser.parse_args()

# Carrega varáveis da linha de comando
TEST_SIZE = float(args.test_size)
K = int(args.kneighbours)
METRIC = int(args.metric)

# Carregar o dataset MNIST
mnist = load_digits()
X = mnist.data
y = mnist.target

# Separa os conjuntos de teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# Treina o classificador
knn = KNeighborsClassifier(n_neighbors=K, p=METRIC)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))