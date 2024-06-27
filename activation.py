import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A
    