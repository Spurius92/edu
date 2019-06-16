'''
Here are implementations of  important functions for machine learning in numpy
'''

import numpy as np


class Sigmoid:
    def __init__(self):
        self.state = None

    def forward(self, x):
        self.state = 1 / (1 + np.exp(-x))

    def derivative(self):
        return self.state * (1 - self.state)


class Tanh:
    def __init__(self):
        self.state = None

    def forward(self, x):
        self.state = np.tanh(x)

    def derivative(self):
        return 1 - self.state ** 2


class ReLU:
    def __init__(self):
        self.state = None

    def forward(self, x):
        self.state = x * (x > 0)

    def derivative(self):
        '''
        Derivative of ReLU is simply 0, if x <= 0, and it is 1 everywhere else.
        Don't forget to specify data type, if neccessary'''
        return (self.state > 0)


# class SoftmaxCrossEnropy:
#     def __init__(self):
#         self.logits = None
#         self.labels = None
#         self.loss = None

#         exponents = np.exp(x)
#         return  exponents / np.sum(exponents)


class Log_softmax:
    def __init__(self):
        self.state = None

    def forward(self, x):
        self.state = x - x.exp().sum(-1).log().unsqueeze(-1)


def cross_entropy(X, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    Note that y is not one-hot encoded vector.
    It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = softmax(X)
    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss


def delta_cross_entropy(X, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    Note that y is not one-hot encoded vector.
    It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m), y] -= 1
    grad = grad/m
    return grad
