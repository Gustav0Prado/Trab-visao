#!/usr/bin/python3

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import argparse

parser = argparse.ArgumentParser(
                    prog='Programa de testes usando MNIST',
                    description='Realiza teste no dataset MNIST usando os parametros passados')

parser.add_argument('kneighbours') # Quantidade de neighbours do kNN, num > 0
parser.add_argument('metric') # Metrica pode ser distância de Manhattan (1) ou distância Euclidiana (2)
parser.add_argument('test_size') # Porcentagem do conjunto destinada ao teste
parser.add_argument('val_size') # Porcentagem do conjunto destinada a validação

args = parser.parse_args()

# Carrega varáveis da linha de comando
TEST_SIZE = float(args.test_size)
VAL_SIZE = float(args.val_size)
K = int(args.kneighbours)
METRIC = int(args.metric)

# Carregar o dataset MNIST
mnist = load_digits()
X = mnist.data
y = mnist.target

# Separa os conjuntos de teste e treino
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, random_state=42)

# Treina o classificador
knn = KNeighborsClassifier(n_neighbors=K, p=METRIC)
knn.fit(X_train, y_train)

y_val_pred = knn.predict(X_val)
y_pred = knn.predict(X_test)

print("Validação: ", accuracy_score(y_val, y_val_pred))
print("Teste:     ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))