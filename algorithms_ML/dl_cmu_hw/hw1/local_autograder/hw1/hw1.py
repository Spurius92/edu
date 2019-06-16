"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1 / (1 + np.exp(-x))
        # return self.state

    def derivative(self):
        return self.state * (1 - self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)

    def derivative(self):
        return 1 - self.state ** 2


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = x * (x > 0)
        # return self.state

    def derivative(self):
        
        return (self.state > 0).astype(np.float64)


# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        self.logits = x
        self.labels = y
        b = np.max(self.logits, axis=1, keepdims=True)
        exp_scores = np.exp(self.logits - b)
        self.sm = exp_scores / np.sum(exp_scores)

        self.loss = b + np.log(np.sum(exp_scores, axis=1, keepdims=True))
 

    def derivative(self):
        pass

        


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        # if eval:
        #something here

        self.x = x

        self.mean = 1. / self.x.shape[0] * np.sum(self.x, axis = 0)
        self.var = 1. / self.x.shape[0] * np.sum((self.x - self.mean) ** 2, axis = 0)
        self.norm == (self.x - self.mean) / np.sqrt(self.var + self.eps)
        self.out = self.gamma * self.norm + self.beta 

        # update running batch statistics
        self.running_mean = self.alpha * self.mean + (1 - self.alpha) * self.running_mean
        self.running_var = self.alpha * self.var + (1 - self.alpha) * self.running_var

    def backward(self, delta):

# self.dbeta = np.sum(dy, axis=0)
# self.dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=0)
# dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dy - np.sum(dy, axis=0)
#     - (h - mu) * (var + eps)**(-1.0) * np.sum(dy * (h - mu), axis=0))

# def batchnorm_backward_alt(dout, cache):
#   gamma, xhat, istd = cache
#   N, _ = dout.shape

#   dbeta = np.sum(dout, axis=0)
#   dgamma = np.sum(xhat * dout, axis=0)
#   dx = (gamma*istd/N) * (N*dout - xhat*dgamma - dbeta)

#   return dx, dgamma, dbeta
        raise NotImplemented


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.randn(d0, d1)


def zeros_bias_init(d):
    return np.zeros(d)


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.W = weight_init_fn(input_size, hiddens)
        self.dW = None
        self.b = bias_init_fn(hiddens, output_size)
        self.db = None

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = 0


        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.layers = [hiddens] 
        # activations = [i for i in self.activations]
        self.z = None
        self.y = None
        self.grads = []

    def forward(self, x):

        z = x @ self.W + self.b
        
        y_hat = self.activations(z)
        
        self.criterion = criterion()

        self.loss = criterion(y, y_hat)

        
    def zero_grads(self):
        self.grads.clear()

    def step(self):
        for idx in range(self.nlayers):
            print('idx:', idx)
            self.W -= self.lr * self.dW
            self.b -= self.lr * self.db

    def backward(self, labels):
        self.loss = self.criterion.forward(self.layers[-1], labels)
        print('Err---------')
        print('Wx shape:', self.layers[-1].shape)
        print('W shape:', self.W[-1].shape)

        grad_loss = self.criterion.derivative()
        print('Grad loss:', grad_loss)

        dinput = np.dot(self.W[-1], grad_loss.T)
        print('dinput', dinput)

        self.dW = np.dot()



    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):

            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    for b in range(0, len(testx), batch_size):

        pass  # Remove this line when you start implementing this
        # Test ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented
