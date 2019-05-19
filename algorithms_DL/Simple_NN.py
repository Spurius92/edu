class SimpleNet:
    def __init__(self, input_size, output_size):
        '''
        Simple one layer net. Here are the initial parameters W and b.
        W is weight matrix of shape input*output.  Xavier initialization here
        b is a bias. It has shape as output
        and it is full of zeros at the beginning
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.W = 2 / (len(self.input_size) ** 0.5) * np.random.randn(self.input_size, self.output_size)
        self.b = np.zeros(len(self.output_size))

    def forward(self, X):
        W = self.W
        b = self.b
        N, D = X.shape

        y_1 = np.dot(X, W) + b # dot product
        y[y < 0] = 0  # ReLU 

        

        
