import numpy as np
from activation import sigmoid, leakyRelu

class SigmoidLayer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
    def forward_prop(self, inputs):
        z = np.dot(inputs, self.weights) + self.biases
        self.out = sigmoid(z)

class ReluLayer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
    def forward_prop(self, inputs):
        z = np.dot(inputs, self.weights) + self.biases
        self.out = leakyRelu(z)