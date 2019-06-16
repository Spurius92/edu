import numpy as np


def accuracy(output, target):
    pred = np.max(output, axis=1)
    return (pred == target).float().mean()
