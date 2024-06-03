# MNIST Neural Network with NumPy

- [Network Structure](#network-structure)
- [Activation Functions](#activation-functions)
- [Loss Function](#loss-function)
- [Back Propagation](#back-propagation)
- [He Initialization](#he-initialization)
- [Analyzing Results](#analyzing-results)

## Network Structure

Network Structure

## Activation Functions

### ReLU

$$
ReLU(z) =
  \begin{cases}
    0 &\text{if } z \le 0 \\
    z &\text{if } z > 0 \\
  \end{cases}
$$

### Softmax

$$
softmax(z_k) = \frac{e^{z_k}}{\sum_i e^{z_i}}
$$

## Loss Function

### Categorical Cross Entropy Loss

$$
Loss = -\sum_{i=1}^H y_i \cdot \log \hat{y}
$$

Where $H$ is the number of 'categories', $y$ the true label and $\hat{y}$ the predicted label.
## Back Propagation

Back Propagation

## He Initialization

He Initialization

## Analysing Results

Analysing results
