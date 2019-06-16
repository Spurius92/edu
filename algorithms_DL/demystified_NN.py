import numpy as np
from scipy import optimize

'''
implemetnation of neural network from Welch labs YouTube tutorial "Neural Networks Demystified"

'''


class Neural_Network(object):
    def __init__(self):
        # define hyperparameters
        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros(self.ouput_size)

    def forward(self, X):
        # forward propagation
        self.z2 = np.dot(X, self.W1) + self.b1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2) + self.b2
        y_hat = self.sigmoid(self.z3)
        return y_hat

    def sigmoid(self, z):
        # sigmoid activation funciton
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        # derivative of sigmoid function
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def loss_function(self, X, y):
        # simple MSE
        self.y_hat = self.forward(X)
        J = 0.5 * np.sum((y - self.y_hat) ** 2)
        return J

    def loss_function_prime(self, X, y):
        # compute derivative with respect to w1 and w2
        self.y_hat = self.forward(X)

        delta3 = np.multiply(- (y - self.y_hat), self.sigmoid_prime(self.z3))
        dJ_dW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_prime(self.z2)
        dJ_dW1 = np.dot(X.T, delta2)

        return dJ_dW1, dJ_dW2

    def get_params(self):
        # get W1 and W2 rolled into a vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def set_params(self, params):
        # set W1 and W2 using single parameter vector
        W1_start = 0
        W1_end = self.hidden_size * self.input_size
        self.W1 = np.reshape(params[W1_start:W1_end], (self.input_size, self.hidden_size))

        W2_end = W1_end + self.hidden_size * self.output_size
        self.W2 = np.reshape(params[W1_end: W2_end], (self.hidden_size, self.output_size))

    def compute_gradients(self, X, y):
        dJ_dW1, dJ_dW2 = self.loss_function_prime(X, y)
        return np.concatenate((dJ_dW1.ravel(), dJ_dW2.ravel()))


'''
compute gradient numerically, like simple limit of the function
'''


def compute_numerical_gradient(N, X, y):
    params_initial = N.get_params()
    numgrad = np.zeros(params_initial.shape)
    perturb = np.zeros(params_initial.shape)
    epsilon = 1e-4

    for p in range(len(params_initial)):
        # set perturbation vector
        perturb[p] = epsilon
        N.set_params(params_initial + perturb)
        loss2 = N.loss_function(X, y)

        N.set_params(params_initial - perturb)
        loss1 = N.loss_function(X, y)

        # compute numerical gradient - slope between values
        numgrad[p] = (loss2 - loss1) / (2 * epsilon)

        # Return the values we changed back to zero
        perturb[p] = 0

    N.set_params(params_initial)

    return numgrad


'''
new trainer class

BFGS (Quasi-Newton) optimization method.
Named after Broyden_Fletcher-Goldfarb-Shanno.

'''


class Trainer(object):
    def __init__(self, N):
        # make a local reference to the network
        self.N = N

    def loss_function_wrapper(self, params, X, y):
        self.N.set_params(params)
        loss = self.N.loss_function(X, y)
        grad = self.N.comput_gradients(X, y)
        return loss, grad

    def callback(self, params):
        self.N.set_params(params)
        self.J.append(self.N.loss_function(self.X, self.y))

    def train(self, X, y):
        # make internal variable for callback function
        self.X = X
        self.y = y

        # empty list to store losses
        self.J = []
        params0 = self.N.get_params()
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.loss_function_wrapper, params0, jac=True, method='BFGS',
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.set_params(_res.X)
        self.optimization_results = _res


'''
Regularization part here
'''

Lambda = 1e-4


# Need to make changes to loss function and loss_function_prime
def loss_function(self, X, y):
    # compute loss for given X and y. Use weights already stored in class

    self.y_hat = self.forward(X)
    J = 0.5 * sum((y - self.y_hat) ** 2) / X.shape[0] + (self.Lambda/2) * (sum(self.W1 ** 2) + sum(self.W2 ** 2))
    return J


def loss_function_prime(self, X, y):
    # compute derivative with respect to W1 and W2 for a given X and y
    self.y_hat = self.forward(X)
    delta3 = np.multiply(-(y - self.y_hat), self.sigmoid_prime(self.z3))
    # Add gradient of regularization term:
    dJ_dW2 = np.dot(self.a2.T, delta3) + self.Lambda * self.W2

    delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_prime(self.z2)
    # Add gradient of regularization term
    dJ_dW1 = np.dot(X.T, delta2) + self.Lambda * self.W1

    return dJ_dW1, dJ_dW2
