import numpy as np

class Layer:
    """
    Represents a neural network layer.

    Attributes
    ----------
    units : int
        The number of units (neurons) in the layer.
    W : ndarray or None
        The weights of the layer.
    b : ndarray or None
        The biases of the layer.
    A : ndarray or None
        The output of the activation function, applied to Z, computed during forward propagation.
    Z : ndarray or None
        The linear combination output (before activation) of the layer.
    vW : ndarray or None
        The velocity of weights for SGD with momentum.
    vb : ndarray or None
        The velocity of biases for SGD with momentum.
    dZ : ndarray or None
        The gradient of the cost with respect to Z.
    dW : ndarray or None
        The gradient of the cost with respect to W.
    db : ndarray or None
        The gradient of the cost with respect to b.
    """

    def __init__(self, units):
        """
        Parameters
        ----------
        units : int
            The number of units (neurons) in the layer.

        """
            
        self.units = units
        
        self.W = None
        self.b = None
        
        self.A = None
        self.Z = None

        self.vW = None
        self.vb = None
        
        self.dZ = None
        self.dW = None
        self.db = None

    def init_weights_and_biases(self, input_units):
        self.W = np.random.randn(self.units, input_units) * np.sqrt(2.0 / input_units)
        self.b = np.zeros((self.units, 1))

    def init_velocities(self):
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
    
    def dense(self, a_in, W, b, activation):
        z = W.dot(a_in) + b
        a_out = activation(z)
        return z, a_out

    def update_weights_biases(self, alpha, momentum_applied):
        if(momentum_applied):
            self.W -= alpha * self.vW
            self.b -= alpha * self.vb
        else:
            self.W -= alpha * self.dW
            self.b -= alpha * self.db

        return self.W, self.b
    
    def update_velocities(self, beta):
        self.vW = beta * self.vW + (1 - beta) * self.dW
        self.vb = beta * self.vb + (1 - beta) * self.db

        return self.vW, self.vb
    