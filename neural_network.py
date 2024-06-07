import numpy as np
from mnist_data_handler import MnistDataHandler
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers):

        self.layers = layers
        self.loss = []

        self.W1 = []
        self.W2 = []
        self.W3 = []

        self.b1 = []
        self.b2 = []
        self.b3 = []

        self.vW1 = []
        self.vW2 = []
        self.vW3 = []

        self.vb1 = []
        self.vb2 = []
        self.vb3 = []

    def init_weights_and_biases(self, X, layers):
        # He initialisation
        W1 = np.random.randn(layers[0], X.shape[0]) * np.sqrt(2.0/X.shape[0])
        W2 = np.random.randn(layers[1], layers[0]) * np.sqrt(2.0/layers[0])
        W3 = np.random.randn(layers[2], layers[1] ) * np.sqrt(2.0/layers[1])

        b1 = np.zeros((layers[0], 1))
        b2 = np.zeros((layers[1], 1))
        b3 = np.zeros((layers[2], 1))

        return W1, W2, W3, b1, b2, b3
    
    def init_velocities(self, W1, W2, W3, b1, b2, b3):

        vW1 = np.zeros_like(W1)
        vW2 = np.zeros_like(W2)
        vW3 = np.zeros_like(W3)

        vb1 = np.zeros_like(b1)
        vb2 = np.zeros_like(b2)
        vb3 = np.zeros_like(b3)

        return vW1, vW2, vW3, vb1, vb2, vb3

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        return A
    
    def one_hot(self, Y):
        return np.array([i == Y for i in range(10)])
    
    def categorical_cross_entropy(self, Y, Y_hat):
        # Avoid numerical instability by clipping values
        Y_hat = np.clip(Y_hat, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(Y * np.log(Y_hat), axis=1))

    def predictions(self, A2):
        return np.argmax(A2, axis=0)

    def incorrect_predictions(self, X, Y, Y_hat):
        incorrect_images = []
        incorrect_labels = []
        
        for i in range(Y.size):
            if(Y[i] != Y_hat[i]):
                incorrect_images.append(X[:, i])
                incorrect_labels.append(Y_hat[i])

        return incorrect_images, incorrect_labels
                
    def accuracy(self, Y, Y_hat):
        return np.mean(Y == Y_hat)
    
    def plot_loss(self):
        plt.plot(self.loss, color = 'r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs epochs')
        plt.show()

    def dense(self, a_in, W, b, activation):
        z = W.dot(a_in) + b
        if(activation == 'relu'):   
            a_out = self.relu(z)
        elif(activation == 'softmax'):
            a_out = self.softmax(z)
        return z, a_out

    def forward_prop(self, X, W1, b1, W2, b2, W3, b3):
        z1, a1 = self.dense(X, W1, b1, 'relu')
        z2, a2 = self.dense(a1, W2, b2, 'relu')
        z3, a3 = self.dense(a2, W3, b3, 'softmax')
        
        return a1, a2, a3, z1, z2, z3
    
    def back_prop(self, Z1, A1, Z2, A2, W2, A3, W3, X, Y):
        m = Y.size

        dZ3 = A3 - self.one_hot(Y)
        dW3 = 1/m * dZ3.dot(A2.T)
        db3 = 1/m * np.sum(dZ3, 1).reshape(-1, 1)

        dZ2 = W3.T.dot(dZ3) * self.relu_derivative(Z2)
        dW2 = 1/m * dZ2.dot(A1.T)
        db2 = 1/m * np.sum(dZ2, 1).reshape(-1, 1)

        dZ1 = W2.T.dot(dZ2) * self.relu_derivative(Z1)
        dW1 = 1/m * dZ1.dot(X.T)
        db1 = 1/m * np.sum(dZ1, 1).reshape(-1, 1)

        return dW1, db1, dW2, db2, dW3, db3

    def update_params(self, W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
        W1 -= alpha * dW1
        W2 -= alpha * dW2
        W3 -= alpha * dW3

        b1 -= alpha * db1
        b2 -= alpha * db2
        b3 -= alpha * db3

        return W1, W2, W3, b1, b2, b3
    
    def update_params_with_velocity(self, W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
        W1 = W1 - alpha * dW1
        W2 = W2 - alpha * dW2
        W3 = W3 - alpha * dW3

        b1 = b1 - alpha * db1
        b2 = b2 - alpha * db2
        b3 = b3 - alpha * db3 

        return W1, W2, W3, b1, b2, b3
    
    def update_velocity_params(self, vW1, vb1, vW2, vb2, vW3, vb3, dW1, db1, dW2, db2, dW3, db3, beta1):
        vW1 = beta1 * vW1 + (1 - beta1) * dW1
        vW2 = beta1 * vW2 + (1 - beta1) * dW2
        vW3 = beta1 * vW3 + (1 - beta1) * dW3

        vb1 = beta1 * vb1 + (1 - beta1) * db1
        vb2 = beta1 * vb2 + (1 - beta1) * db2
        vb3 = beta1 * vb3 + (1 - beta1) * db3
        
        return vW1, vW2, vW3, vb1, vb2, vb3

    def training_loop(self, X, Y, epochs, alpha, momentum_applied):
        W1, W2, W3, b1, b2, b3 = self.init_weights_and_biases(X, self.layers)
        if(momentum_applied):
            vW1, vW2, vW3, vb1, vb2, vb3 = self.init_velocities(W1, W2, W3, b1, b2, b3)

        for i in range(epochs):
            a1, a2, a3, z1, z2, _ = self.forward_prop(X, W1, b1, W2, b2, W3, b3)
            dW1, db1, dW2, db2, dW3, db3 = self.back_prop(z1, a1, z2, a2, W2, a3, W3, X, Y)

            if (momentum_applied):
                vW1, vW2, vW3, vb1, vb2, vb3 = self.update_velocity_params(vW1, vb1, vW2, vb2, vW3, vb3, dW1, db1, dW2, db2, dW3, db3, beta1=0.9)
                W1, W2, W3, b1, b2, b3 = self.update_params_with_velocity(W1, b1, W2, b2, W3, b3, vW1, vb1, vW2, vb2, vW3, vb3, alpha)
            else:
                W1, W2, W3, b1, b2, b3 = self.update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

            self.loss.append(self.categorical_cross_entropy(self.one_hot(Y), a3))
            if(i % 10 == 0):
                print('Epoch: ', i)
                print('Accuracy: ', self.accuracy(Y, self.predictions(a3)))
                print('Loss: ', self.categorical_cross_entropy(self.one_hot(Y), a3))
        print('Accuracy after training: ', self.accuracy(Y, self.predictions(a3)))
       
        # save the post-training weights and biases
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.b1 = b1
        self.b2 = b2 
        self.b3 = b3

    def run_testing(self, X, Y, W1, W2, W3, b1, b2, b3):
        _, _, a3, _, _, _ = self.forward_prop(X, W1, b1, W2, b2, W3, b3)
        print('Test Accuracy: ', self.accuracy(Y, self.predictions(a3)))

def main():
    mnist_data_handler = MnistDataHandler()
    x_train, y_train = mnist_data_handler.load_training_data()

    x_train = x_train.T
    x_train = mnist_data_handler.normalise(x_train)

    neural_network = NeuralNetwork(layers=[40, 20, 10])
    neural_network.training_loop(x_train, y_train, epochs=200, alpha=0.5, momentum_applied=True)

    x_test, y_test = mnist_data_handler.load_test_data()
    x_test = x_test.T
    x_test = mnist_data_handler.normalise(x_test)

    neural_network.run_testing(x_test, y_test,
                                neural_network.W1, neural_network.W2, neural_network.W3,
                                  neural_network.b1, neural_network.b2, neural_network.b3)
    
    neural_network.plot_loss()

    plt.show()

    
if __name__ == '__main__':
    main()