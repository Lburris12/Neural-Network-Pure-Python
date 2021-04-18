from layer import SigmoidLayer, ReluLayer
from activation import sigmoidDerivative, leakyReluDerivative
import numpy as np

def convert_labels(labels):
    variance = set(labels)
    new_labels = [[0 for i in range(len(variance))] for j in range(len(labels))]
    for i, label in enumerate(labels):
        new_labels[i][label] = 1
    
    return new_labels

class NeuralNetwork:
    def __init__(self, layer_details, activation_function):
        self.activation_function = activation_function

        if self.activation_function == 'sigmoid':
            self.layers = [SigmoidLayer(n_inputs, n_neurons) for n_inputs, n_neurons in layer_details]

        elif self.activation_function == 'relu':
            self.layers = [ReluLayer(n_inputs, n_neurons) for n_inputs, n_neurons in layer_details]

    def forward_prop(self, inputs):
        out = inputs
        for layer in self.layers:
            layer.forward_prop(out)
            out = layer.out
        self.out = out
    
    def back_prop(self, labels, learning_rate):
        alter_values = []
        prev_layer = None
        prev_dc_dz = None
        for i, layer in enumerate(reversed(self.layers[1:])):
            if self.activation_function == 'relu':
                func_derivative = leakyReluDerivative(layer.out)
            elif self.activation_function == 'sigmoid':
                func_derivative = sigmoidDerivative(layer.out)
            
            if i == 0:
                dc_dz = np.multiply(2 * (layer.out-labels), func_derivative)
            else:
                dc_dcl = np.sum(prev_dc_dz * prev_layer.weights, axis=1)
                dc_dz = dc_dcl * func_derivative
            dc_dw = np.dot(self.layers[-(i+2)].out.T, dc_dz)
            alter_values.insert(0, [])
            alter_values[0].append(learning_rate * dc_dw)
            alter_values[0].append(learning_rate * dc_dz)
            prev_dc_dz = dc_dz
            prev_layer = layer
        
        for values, layer in zip(alter_values, self.layers[1:]):
            old = layer.weights
            layer.weights -= values[0]
            layer.biases -= values[1]
          
    
    def fit(self, inputs, labels, learning_rate, iters):
        labels = convert_labels(labels)

        length = len(inputs)
        for i in range(iters):
            for j in range(length):
                self.forward_prop(inputs[j])
                self.back_prop(labels[j], learning_rate)
    
    def predict(self, inputs):
        self.forward_prop(inputs)
        self.prediction = np.argmax(self.out)
    
    def test(self, inputs, labels):
        correct = 0
        for i in range(len(inputs)):
            self.predict(inputs[i])
            if self.prediction == labels[i]:
                correct += 1
        self.accuracy = str(correct/len(inputs))[:4]
