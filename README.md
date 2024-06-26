<div align="center">

# MNIST Neural Network with NumPy

A Neural Network, trained on the MNIST database, with the task of classifying handwritten digits. Made without the use of ML libraries such as TensorFlow or Pytorch, only using NumPy. Currently runs with a 98% Accuracy!

</div>

## Contents ##
&rightarrow; [MNIST Database](#mnist-database)\
&rightarrow; [Overview on How the Nework Learns](#overview-on-how-the-network-learns)\
&rightarrow; [Network Structure](#network-structure)\
&rightarrow; [Activation Functions](#activation-functions)\
&rightarrow; [One-Hot Encoding](#one-hot-encoding)\
&rightarrow; [Loss Function](#loss-function)\
&rightarrow; [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)\
&rightarrow; [Gradient Descent with Momentum](#gradient-descent-with-momentum)\
&rightarrow; [Back Propagation](#back-propagation)\
&rightarrow; [He Initialisation](#he-initialisation)\
&rightarrow; [Analysing Results](#analysing-results)



## Mnist Database
The MNIST database is a large database, containing 70,000 images of handwritten digits.

Each image is black and white, 28x28 pixels in size, and contains a singular handwritten digit.  

![MNIST Example Images](assets/MNIST_Examples.png)

For the purpose of this network, 60,000 images are reserved for training and the remaining 10,000 are used for testing the network on unseen images.

## Overview on How the Network Learns

1. **Initialisation**: The networks weights and biases are initialised.
3. **Forward Propagation**: The input data is passed through the network and predictions are obtained.
4. **Computing Loss**: The predictions are passed through the loss function to measure the accuracy of the network.
5. **Backward Propagation**: Using partial derivatives, compute the gradient of the loss function with respect to each weight and bias in the network.
6. **Update Parameters**: Update the weights and biases, opposing the direction of the gradient, to reach a point of minimum loss.
7. **Repeat steps 2-5**.


## Network Structure
This image is a simplified version of the networks architecture. It contains an input layer, hidden layers and an output layer. One important thing to note is that the output contains 10 different nodes. These correspond to the 10 digits (0-9) that the network is attempting to classify.

The ReLU functions in the hidden layers introduce non-linearity and a softmax function is applied to the output to convert the raw output (logits) to probabilities, which sum to 1.

In reality, the input layer contains 784 (28*28) input features and there are many more neurons in the hidden layers, however the general architecture is the same between the neural network in this image and the one in the code.

![Neural Network Architecture Image](assets/neural_network_architecture.png)

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
Softmax(z_i) = \frac{e^{z_i}}{\sum_k e^{z_k}}
$$

## One-Hot Encoding
When representing the categorical output data $y$ we can use one-hot encoding. This involves representing the data as a binary vector.

For example, we can encode the number $3$ as 

$$
\[0,0,0,1,0,0,0,0,0,0\]
$$

Here, each number (0-9) can be represented by simply setting it's respective index (starting from 0) to 1 and all others to 0. This works particularly well when paired with the softmax and categorical cross-entropy loss function.

## Loss Function

### Categorical Cross-Entropy Loss

$$
L = -\sum_{i=1}^H y_i \cdot \log \hat{y_i}
$$

Where $H$ is the number of 'categories', $y$ the true label and $\hat{y}$ the predicted label.

## Mini-Batch Gradient Descent

Mini-batch gradient descent is a variation of the gradient descent algorithm that splits the training dataset into small batches. When data is processed in these small batches it means that the weights and biases are updated with each mini-batch, unlike batch gradient descent where the traning set is processed as a whole unit.

## Gradient Descent with Momentum

Gradient Descent with Momentum is an optimisation technique which allows for the network to converge faster to the optimal solution. This is done by calculating an exponentially weighted average of the gradients and then this averaged gradient is used to update the weights and biases.

**Velocity Parameter Update**

$$
\begin{align*}
V_{\delta w} &:= \beta \cdot V_{\delta w} + (1-\beta)\cdot \delta w\\
V_{\delta b} &:= \beta \cdot V_{\delta b} + (1-\beta)\cdot \delta b
\end{align*}
$$

**Weight and Bias Update**

$$
\begin{align*}
w &:= w-\alpha V_{\delta w}\\
b &:= b-\alpha V_{\delta b}
\end{align*}
$$

## Back Propagation

### Deriving the derivative of the ouput layer $\left(\frac{\delta L}{\delta z_k}\right)$

By the previously stated definition of Categorical Cross Entropy Loss

$$
L = -\sum_{i=1} y_i \log(o_i)
$$

where $o_i$ is the output of the softmax function, given by

$$
o_i = \frac{e^{z_i}}{\sum_{k=1} e^{z_k}}
$$

Now, using partial derivatives

$$
\begin{align*}
\frac{\delta L}{\delta z_k} &= \frac{\delta(-\sum_{i=1}y_i\log(o_i))}{\delta(z_k)}\\
&= -\sum_{i=1} \frac{\delta(y_i \log(o_i))}{\delta(z_k)}\\
&= -\sum_{i=1} y_i\cdot \frac{\delta(\log(o_i)}{\delta(z_k)}\hspace{10pt} \text{Since } y_i \text{ is independent of } z_k
\end{align*}
$$

By the chain rule we find that

$$
\frac{\delta{\log(o_i)}}{\delta{z_k}} = \frac{\delta(\log(o_i))}{\delta(o_i)}\cdot \frac{\delta(o_i)}{\delta(z_k)}
$$

Substituting this back in

$$
\frac{\delta L}{\delta z_k} = -\sum_{i=1} \left[ y_i \cdot \frac{\delta(\log(o_i))}{\delta(o_i)}\cdot \frac{\delta(o_i)}{\delta(z_k)}\right]
$$

Since $\frac{\delta(\log(o_i))}{\delta(o_i)} = \frac{1}{o_i}$

$$
\frac{\delta L}{\delta z_k} = -\sum_{i=1} \left[ \frac{y_i}{o_i} \cdot \frac{\delta(o_i)}{\delta(z_k)}\right]
$$

$\frac{\delta o_i}{\delta z_k}$ is the derivative of the Softmax output with respect to the input z_k which is given by

$$
\frac{\delta o_i}{\delta z_k} = 
\begin{cases}
    o_k(1-o_k) &\text{if } i = k \\
    -o_i \cdot o_k &\text{if } i \ne k \\
  \end{cases}
$$

In order to use this definition within our equation for $\frac{\delta L}{\delta z_k}$ we must re-write the sum

$$
\begin{align*}
\frac{\delta L}{\delta z_k} &= -\left[ \sum_{i\ne k}\left(\frac{y_i}{o_k}\cdot -o_i\cdot o_k \right) + \frac{y_k}{o_k}\cdot o_k(1-o_k)\right]\\
&= -\left[ \sum_{i\ne k}\left(-y_i\cdot o_k \right) + y_k(1-o_k)\right]\\
&= -\left[-o_k\sum_{i\ne k}\left(-y_i \right) + y_k(1-o_k)\right]
\end{align*}
$$

Since $y_i$ is one-hot encoded, we know that

$$
\sum_{i=1}y_i = 1 \hspace{10pt} \text{ and } \hspace{10pt} \sum_{i\ne k}y_i = 1 - y_k
$$

Therefore

$$
\begin{align*}
\frac{\delta L}{\delta z_k} &= -\left[-o_k(1-y_k) + y_k(1-o_k)\right]\\
&= -\left[-o_k + o_k\cdot y_k + y_k -o_k\cdot y_k\right]\\
&= o_k - y_k
\end{align*}
$$

## He Initialisation

Weights cannot be initialised to 0. Since $a = \overline{w}X + b =0$ would result in every neuron outputting 0 and all neurons outputting the same value, regardless of input (symmetry).

For this model, He initialisation is used and is defined as follows:

$$
W \sim N(0, \sqrt{\frac{2}{n_{in}}})
$$

This denotes a normal distribution with mean $0$ and standard deviation $\sqrt{\frac{2}{n_{in}}}$ where $n_{in}$ represents the number of input units to the layer.

Using He initialisation reduces the chances of gradients 'vanishing' or 'exploding' during backpropagation and also leads to faster convergence. It is particularly suited for neural networks utilising the ReLU activation function.
## Analysing Results

<div align="center">

| Epochs | Mini-Batch Size | Neurons (Layer1/Layer2/.../Layer n) | Learning Rate | Momentum Applied (True/False) | Accuracy (%) |
|----------|----------|----------|----------|----------|----------|
| 20 | 128 | 200/100/25/10 | 0.5 | True | 98.42 |
| 20 | 128 | 100/50/10 | 0.5 | True | 98.14 |
| 20 | 128 | 100/50/10 | 0.5 | False | 97.51 |
| 20 | 60,000 | 100/50/10 | 0.5 | True | 84.13 |
| 20 | 60,000 | 100/50/10 | 0.5 | False | 65.16 |

</div>

*Note:* A mini-batch size of 60,000 just means that batch gradient descent is being used over mini-batch gradient descent.
