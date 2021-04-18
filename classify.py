import numpy as np
from NN import NeuralNetwork
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

model = NeuralNetwork([[4,4],[4,5],[5,3]], 'sigmoid')
model.fit(x_train, y_train, 0.5, 1000)
model.test(x_test, y_test)

print(model.accuracy)
