import numpy as np
import matplotlib.pyplot as plt

# Creating softmax activation function
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))
    return exp_Z / np.sum(exp_Z, axis = 0, keepdims = True) # Keepdims ensures arrays in subsequent operations are compatible

# tanh activation function
def tanh(x):
    return np.tanh(x)

# Derivative of the hyperbolic tangent
def tanh_derivative(x):
    return 1 - np.tanh(x)**2 

# Loss function: cross-entropy loss
def loss(Y, out):
    N = Y.shape[0]
    error = -np.sum((Y*np.log(out))/N)
    return error

# Feed-forward neural network & loss
def neural_net(X, Y, W1, W2, b1, b2):
    N = X.shape[0]

    Z1 = X.dot(W1) + b1.T
    A1 = tanh(Z1)
    Z2 = A1.dot(W2) + b2.T
    A2 = softmax(Z2.T).T

    loss = loss(Y, A2)
    return A2, A1, loss

 # Gradient calculation
 def neural_net_backward(X, Y, A1, A2, W2):
    N = X.shape[0]
    dZ2 = A2 - Y
    