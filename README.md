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
Softmax(z_i) = \frac{e^{z_i}}{\sum_k e^{z_k}}
$$

## Loss Function

### Categorical Cross Entropy Loss

$$
Loss = -\sum_{i=1}^H y_i \cdot \log \hat{y}
$$

Where $H$ is the number of 'categories', $y$ the true label and $\hat{y}$ the predicted label.
## Back Propagation

### Deriving $\frac{\delta L}{\delta z_k}$

We know that 

$$
L = -\sum_{i=1}^H y_i \log(o_i)
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
&= -\sum_{i=1} y_i\cdot \frac{\delta(\log(o_i)}{\delta(z_k)}\hspace{10pt} \text{Since } y_i \text{ is independent of } o_i
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
    o_i(1-o_i) &\text{if } i = k \\
    -o_i \cdot o_i &\text{if } i \ne k \\
  \end{cases}
$$

In order to use this definition within our equation for $\frac{\delta L}{\delta z_k}$ we must re-write the sum

$$
\begin{align*}
\frac{\delta L}{\delta z_k} &= -\left[ \sum_{i\ne k}\left(\frac{y_i}{o_i}\cdot -o_i\cdot o_i \right) + \frac{y_k}{o_k}\cdot(1-o_k)\right]\\
&= -\left[ \sum_{i\ne k}\left(\frac{y_i}{o_i}\cdot -o_i\cdot o_i \right) + \frac{y_k}{o_k}\cdot(1-o_k)\right]
\end{align*}
$$

## He Initialization



He Initialization

## Analysing Results

Analysing results
