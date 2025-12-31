import numpy as np
from pip._internal import network
from activation_function import *

def init_network():
    network = {}

    # x1, x2 -> s21, s22, s23
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    # s21, s22, s23 - > s31, s32
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    # s31, s32 -> s41, s42
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward_prop(network, x):
    W1 = network['W1']
    b1 = network['b1']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    W2 = network['W2']
    b2 = network['b2']
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    W3 = network['W3']
    b3 = network['b3']
    a3 = np.dot(z2, W3) + b3
    y = identity(a3)

    return y

def test():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward_prop(network, x)
    print(y)

test()