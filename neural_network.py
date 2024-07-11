import numpy as np

from mnist_data_handler import MnistDataHandler
from network_config import NetworkConfig
from layer import Layer
from activation import relu, relu_derivative, softmax
from categorical_cross_entropy import one_hot, categorical_cross_entropy_loss

INPUT_FEATURE_COUNT = 784

class NeuralNetwork:
    def __init__(self, network_config):
        
        hidden_layer_architecture = network_config.layer_architecture

        self.layers = []
        self.layer_count = len(hidden_layer_architecture)

        # initialise the first hidden layer
        first_hidden_layer_unit_count = hidden_layer_architecture[0]
        self.first_hidden_layer = Layer(first_hidden_layer_unit_count)
        self.first_hidden_layer.init_weights_and_biases(INPUT_FEATURE_COUNT)
        if network_config.momentum_applied:
            self.first_hidden_layer.init_velocities()
        self.layers.append(self.first_hidden_layer)
        
        # initialise remaining hidden layers
        for i in range(1, self.layer_count):
            input_units = hidden_layer_architecture[i-1]
            units = hidden_layer_architecture[i]

            hidden_layer = Layer(units)
            hidden_layer.init_weights_and_biases(input_units)
            if network_config.momentum_applied:
                hidden_layer.init_velocities()
            self.layers.append(hidden_layer)

    def forward_prop(self, X):
        
        # propagate the image input X, through the first hidden layer
        z_i, a_i = self.layers[0].dense(X, self.layers[0].W, self.layers[0].b, relu)
        self.layers[0].Z = z_i
        self.layers[0].A = a_i

        # propagate through the remaining hidden layers, up to but not including the output layer
        layer_count = len(self.layers)

        for i in range(1, layer_count-1):

            z_i, a_i = self.layers[i].dense(self.layers[i-1].A, self.layers[i].W, self.layers[i].b, relu)
            self.layers[i].Z = z_i
            self.layers[i].A = a_i

        # propagate through the ouput layer 
        z_i, a_i = self.layers[-1].dense(self.layers[-2].A, self.layers[-1].W, self.layers[-1].b, softmax)
        self.layers[-1].Z = z_i
        self.layers[-1].A = a_i

    def backward_prop(self, X, Y):

        layer_count = len(self.layers)
        m = Y.size

        self.layers[-1].dZ = self.layers[-1].A - one_hot(Y)
        self.layers[-1].dW = 1/m * self.layers[-1].dZ.dot(self.layers[-2].A.T)
        self.layers[-1].db = 1/m * np.sum(self.layers[-1].dZ, 1).reshape(-1, 1)

        for i in range(1, layer_count-1):
            
            W_prev = self.layers[-i].W.T
            dZ_prev = self.layers[-i].dZ
            Z_i = self.layers[-(i+1)].Z

            dZ_i = W_prev.dot(dZ_prev) * relu_derivative(Z_i)
            self.layers[-(i+1)].dZ = dZ_i

            A_prev = self.layers[-(i+2)].A.T
            dW_i = 1/m * dZ_i.dot(A_prev)
            self.layers[-(i+1)].dW = dW_i

            db_i = 1/m * np.sum(dZ_i, 1).reshape(-1, 1)
            self.layers[-(i+1)].db = db_i


        self.layers[0].dZ = self.layers[1].W.T.dot(self.layers[1].dZ) * relu_derivative(self.layers[0].Z)
        self.layers[0].dW = 1/m * self.layers[0].dZ.dot(X.T)
        self.layers[0].db = 1/m * np.sum(self.layers[0].dZ, 1).reshape(-1, 1)
    
    def training_loop(self, X, Y, network_config):

        for i in range(network_config.epochs):

            mini_batches = MnistDataHandler().create_mini_batches(X, Y, network_config.batch_size)

            for mini_batch in mini_batches:
                X_mini, Y_mini = mini_batch

                self.forward_prop(X_mini)
                self.backward_prop(X_mini, Y_mini)

                momentum_applied = network_config.momentum_applied

                for layer in self.layers:
                    if momentum_applied:
                        layer.update_velocities(network_config.beta)
                    layer.update_weights_biases(network_config.alpha, momentum_applied)

            network_output = self.layers[-1].A
            accuracy = self.accuracy(Y_mini, self.predictions(network_output))
            loss = categorical_cross_entropy_loss(one_hot(Y_mini), network_output)

            print(f"Epoch: {i+1}/{network_config.epochs}")
            print('Accuracy: ', accuracy.round(4))
            print('Loss: ', loss.round(4))
            print('------------------------------')
            

    def predictions(self, A_out):
        return np.argmax(A_out, axis=0)

    def accuracy(self, Y, Y_hat):
        total_correct = np.sum(Y == Y_hat)
        return total_correct / Y.size
    
    def run_testing(self, X, Y):
        self.forward_prop(X)
        accuracy = self.accuracy(Y, self.predictions(self.layers[-1].A))
        print('Testing Accuracy: ', accuracy)

def main():
    mnist_data_handler = MnistDataHandler()
    
    network_config = NetworkConfig(
        layer_architecture = [200, 100, 25, 10],
        alpha = 0.5,
        epochs = 20,
        batch_size=128,
        momentum_applied=True
    )
    neural_network = NeuralNetwork(network_config)

    X_train, Y_train = mnist_data_handler.load_training_data()
    X_train = X_train.T
    X_train = mnist_data_handler.normalise(X_train)

    neural_network.training_loop(X_train, Y_train, network_config)

    X_Test, Y_Test = mnist_data_handler.load_test_data()
    X_Test = X_Test.T
    X_Test = mnist_data_handler.normalise(X_Test)

    neural_network.run_testing(X_Test, Y_Test)

if __name__ == "__main__":
    main()