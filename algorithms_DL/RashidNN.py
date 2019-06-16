import numpy as np
from math import exp

'''
Veeery simple neural net from Tarik Rashid's book 'making your own neural network'
'''


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.w1 = np.random.normal(0.0, np.power(hidden_size, -0.5), (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.normal(0.0, np.power(output_size, -0.5), (hidden_size, output_size))
        self.b2 = np.zeros(output_size)
        self.lr = lr
        self.activation = lambda x: exp(x)

    def forward(self, x, y):
        inputs = np.array(x, ndmin=2)
        labels = np.array(y, ndmin=2)

        self.z1 = np.dot(inputs, self.w1)
        self.y1 = self.activation(self.z1)

        self.z2 = np.dot(self.y1, self.w2)
        self.y2 = self.activation(self.z2)

        self.error = labels - self.y2

    def backward(self):
        self.dw2 = np.dot(self.w2.T, self.error)
        self.w2 += self.lr * np.dot(self.error, (self.y2 * (1 - self.y2)))

        self.w1 += self.lr * np.dot(self.dw2, (self.y1 * (1 - self.y1)))

    def __call__(self, x, y):
        return self.forward(x, y)
