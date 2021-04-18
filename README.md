# Neural-Network-Pure-Python
created using python and numpy

## Description
It is a simple multi-layer perception (MLP) that can be used with the sigmoid activation function or the leakyReLU activation function. The backpropagation algorithim utilizes the sgd algorithm with a squared error loss.  I used the leakyRelu instead of the regular ReLU function in order to reduce backpropagation difficulties.

## How To Use
In this project you will find four files: NN.py, layer.py, activation.py, and classify.py. The "NN.py" file is the neural network class that utilizies the "layer.py" file to create its layers. "layer.py" contains layer classes which utilize "activation.py" for their activation functions. "classify.py" is just an example of the neural network in action using the sklearn iris dataset with the parameters I found worked best.

after importing "NN.py", initializing a model looks something like this:

`model = NeuralNetwork([[3,3],[3,8],[8,4]], 'sigmoid')`

Basically we're creating a neural network with three layers:
- input layer has 3 inputs with 3 neurons
- hidden layer has 3 inputs with 8 neurons
- output layer has 8 inputs with 4 neurons
- each layer with the sigmoid activation function

*Note: within your layer details, for each layer the input must match the previous layer's nuerons count. Also for classification, the output layer's neuron count must match the amount of classes there are*

Training the model looks like this:

`model.fit(X,Y, 0.1, 500)`

Where "X" is all your data, "Y" is all your labels, 0.1 is the learning rate, and 500 is the epochs or how many times you want the NN to train on the data. The labels should be a list of integers specifing what class the example is.

To test your model looks like this:

`model.test(X, Y)`

Where "X" is all your test data, and "Y" is all your test labels.

After you have tested your model you can check the accuracy of it by running:

`print(model.accuracy)`

*Note: this will only work if you have run "model.test" previously*

