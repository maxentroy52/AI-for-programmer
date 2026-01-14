import numpy as np

def step(x):
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity(x):
    return x

def softmax(x):
    x_max = np.max(x)
    x_new = x - x_max

    x_new_exp = np.exp(x_new)
    x_new_exp_sum = np.sum(x_new_exp)
    return x_new_exp / x_new_exp_sum