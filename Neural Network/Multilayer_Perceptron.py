#Multilayer perceptron
#Feed forward neural network

#modified from 
#https://towardsdatascience.com/creating-neural-networks-from-scratch-in-python-6f02b5dd911
#https://colab.research.google.com/drive/1ku3LvrqovKOzeCTkW6bLHYxNKMaG4KFv

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

# Activation Functions
def relu(x):
    return np.maximum(x, 0)

def d_relu(x):
    return np.heaviside(x, 1)
    
def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.square(np.tanh(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)

# Loss Functions
def logloss(y, a):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))

def d_logloss(y, a):
    return (a - y) / (a * (1 - a))

def mse(y, a):
    return (y - a)**2

def d_mse(y, a):
    return -2 * (y - a)


def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo / expo_sum


# The layer class
class Layer:
    activationFunctions = {
        'tanh': (tanh, d_tanh),
        'sigmoid': (sigmoid, d_sigmoid),
        'relu': (relu, d_relu)
    }
    learning_rate = 0.01

    def __init__(self, inputs, neurons, activation):
        self.W = np.random.randn(neurons, inputs)
        self.b = np.zeros((neurons, 1))
        self.act, self.d_act = self.activationFunctions.get(activation)

    def feedforward(self, x_prev):
        self.x_prev = x_prev
        self.Z = np.dot(self.W, self.x_prev) + self.b
        self.y = self.act(self.Z)
        return self.y
    
    def backprop(self, dy):
        dZ = np.multiply(self.d_act(self.Z), dy)
        dW = np.dot(dZ, self.x_prev.T)
        db = dZ
        dA_prev = np.dot(self.W.T, dZ)

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return dA_prev


#XOR data

x_train = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y_train = np.array([[0, 1, 1, 0]])


epochs = 500

inp = x_train.shape[0]
m = x_train.shape[1]

#layers of a neural network
layers = [Layer(inp, 3, 'sigmoid'), Layer(3, 1, 'sigmoid')]
costs = []

for epoch in range(epochs):
    A = x_train
    for layer in layers:
        A = layer.feedforward(A)

    #cost = 1/m * np.sum(logloss(y_train, A))
    cost = np.mean(mse(y_train, A))
    costs.append(cost)

    dA = d_mse(y_train, A)
    for layer in reversed(layers):
        dA = layer.backprop(dA)

# Making predictions
A = x_train
for layer in layers:
    A = layer.feedforward(A)
print(A)

#plotting cost
plt.plot(range(epochs), costs)
plt.show()



