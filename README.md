<div align="center">

# MNIST Neural Network with NumPy

A Neural Network, trained on the MNIST database, with the task of classifying handwritten digits. Made without the use of ML libraries such as TensorFlow or PyTorch, only using NumPy. Currently runs with a 98% Accuracy!

</div>

<div align="center">

View all the details of the project 
[**HERE**](https://wryzxec.github.io/neuralnet.html)

</div>

</br>

## Dependencies
- NumPy : For numerical operations.

### To install NumPy
```bash
pip install numpy
```

## Installation

### To set up and run this project

```bash
git clone https://github.com/wryzxec/MNIST_NeuralNet.git
```

## Initialising the Network
```Python
network_config = NetworkConfig(
        layer_architecture = [200, 100, 25, 10],
        alpha = 0.5,
        beta=0.9,
        epochs = 20,
        batch_size=128,
        momentum_applied=True
    )

neural_network = NeuralNetwork(network_config)
```

## Training the Network
```Python
neural_network.training_loop(X_train, Y_train, network_config)
```

## Testing the Network
```Python
neural_network.run_testing(X_Test, Y_Test)
```


