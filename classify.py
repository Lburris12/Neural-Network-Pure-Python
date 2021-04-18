import numpy as np
from NN import NeuralNetwork
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
#import data

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
#split data into  67% training data and 33% testing data

model = NeuralNetwork([[4,4],[4,5],[5,3]], 'sigmoid')
#initialize model with three layers

model.fit(x_train, y_train, 0.5, 1000)
#train model

model.test(x_test, y_test)
#test model

print(f'accuracy: {model.accuracy}')
