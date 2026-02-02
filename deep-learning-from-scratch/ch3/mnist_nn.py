import sys,os
import numpy as np
import pickle

from tensorboard.backend.event_processing.event_file_inspector import PRINT_SEPARATOR
from tensorflow.python.ops.metrics_impl import accuracy

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from activation_function import *

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = True, one_hot_label = False)
    return x_train, t_train, x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

def main():
    x_train, t_train, x_test, t_test = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x_test)):
        y_pred = predict(network, x_test[i])

        p = np.argmax(y_pred)
        if t_test[i] == p:
            accuracy_cnt += 1

    print(float(accuracy_cnt) / len(x_test))

def print_layout():
    _, _, x, y = get_data()
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']

    print(W1.shape)
    print(W2.shape)
    print(W3.shape)

def batch_predict():
    _, _, x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        y_pred_batch = predict(network, x[i:i + batch_size])
        p = np.argmax(y_pred_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i + batch_size])

    print(float(accuracy_cnt) / len(x))

if __name__ == '__main__':
    # main()
    # print_layout()
    batch_predict()