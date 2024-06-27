import numpy as np

def one_hot(Y):
    return np.array([i == Y for i in range(10)])

def categorical_cross_entropy_loss(Y, Y_hat):
    Y_hat = np.clip(Y_hat, 1e-15, 1 - 1e-15) # Avoid numerical instability by clipping values
    return -np.mean(np.sum(Y * np.log(Y_hat), axis=1))