import numpy as np

def and_perceptron(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1 * x1 + w2 * x2
    if tmp < theta:
        return 0
    else:
        return 1

def add_perceptron_v1(w, b, x):
    tmp = np.sum(w * x) + b
    if (tmp < 0):
        return 0
    else:
        return 1

def test_add_perceptron():
    print(and_perceptron(0, 0))
    print(and_perceptron(0, 1))
    print(and_perceptron(1, 0))
    print(and_perceptron(1, 1))

def test_add_perceptron_v1():
    W = np.array([0.5, 0.5])
    B = -0.7

    X1 = np.array([0, 0])
    X2 = np.array([0, 1])
    X3 = np.array([1, 0])
    X4 = np.array([1, 1])

    Y1 = add_perceptron_v1(W, B, X1)
    Y2 = add_perceptron_v1(W, B, X2)
    Y3 = add_perceptron_v1(W, B, X3)
    Y4 = add_perceptron_v1(W, B, X4)

    print(Y1)
    print(Y2)
    print(Y3)
    print(Y4)

def main():
    test_add_perceptron()
    test_add_perceptron_v1()

main()