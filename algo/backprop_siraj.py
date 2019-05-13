'''
implementation of gradient descent by Siraj.
Very simple, using quadratic loss function
points of data are just one on one regression.
'''
import numpy as np   
# import pandas as pd

def forward_pass(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][0]
        total_error += ( y - (m * x + b)) ** 2
    return total_error / len(points)


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = len(points)

    for i in range(len(points)):
        x = points[i][0]
        y = points[i][0]
        m_gradient += -((2 / N) * x * (y - (m_current * x + b_current)))
        b_gradient += -((2 / N) * (y - (m_current * x + b_current)))
    
    new_b = b_current - learning_rate * b_gradient
    new_m = m_current - learning_rate * m_gradient

    return new_b, new_m

def gradient_descent_runner(points, b_start, m_start, learning_rate, num_iters):
    b = b_start
    m = m_start
    for i in range(num_iters):
        b, m = step_gradient(b, m, points, learning_rate)
    return b, m

def run():
    points = np.genfromtxt('C:\\Users\\Spurius\\Desktop\\algo\\data.csv', delimiter=',')
    print(len(points))
    print(points.shape)
    print(points[:5])
    learning_rate = 0.01
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    # forward_pass(initial_m, initial_b, points)

    print('Starting gradient descent with b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, forward_pass(initial_m, initial_b, points)))
    print('Running...')
    b, m = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print('After {0} iterations b = {1}, m = {2} and error = {3}'.format(num_iterations, b, m, forward_pass(m, b, points)))

if __name__ == '__main__':
    run()