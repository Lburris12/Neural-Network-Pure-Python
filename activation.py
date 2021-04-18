import numpy as np

def sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))
    
def sigmoidDerivative(inputs):
    return inputs * (1 - inputs)

def leakyRelu(inputs):
    return np.maximum(0.3*inputs, inputs)

def leakyReluDerivative(inputs):
    out = []
    for num in inputs[0]:
        if num > 0:
            out.append(1)
        else:
            out.append(0.3)
    return [out]