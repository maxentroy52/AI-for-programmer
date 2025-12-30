import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def naive_test():
    x = np.array([-1, 0, 1])
    y = sigmoid(x)
    print(y)

naive_test()