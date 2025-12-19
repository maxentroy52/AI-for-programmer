import numpy as np

def and_perceptron(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1 * x1 + w2 * x2
    if tmp < theta:
        return 0
    else:
        return 1

def test_add_perceptron():
    print(and_perceptron(0, 0))
    print(and_perceptron(0, 1))
    print(and_perceptron(1, 0))
    print(and_perceptron(1, 1))

def perceptron(w, b, x):
    tmp = np.sum(w * x) + b
    if (tmp < 0):
        return 0
    else:
        return 1

# They have same structure, the only difference between them is the parameter.
and_perceptron_v1 = lambda x: perceptron(np.array([0.5, 0.5]), -0.7, x)
nand_perceptron_v1 = lambda x: perceptron(np.array([-0.5, -0.5]), -0.7, x)
or_perceptron_v1 = lambda x: perceptron(np.array([0.5, 0.5]), 0.3, x)

def xor_perceptron_v1(x):
    s1 = nand_perceptron_v1(x)
    s2 = or_perceptron_v1(x)
    new_input = np.array([s1, s2])
    return and_perceptron_v1(new_input)

def test_all_perceptron():
    print('------------------')
    print(and_perceptron_v1(np.array([1, 1])))
    print(nand_perceptron_v1(np.array([1, 1])))
    print(or_perceptron_v1(np.array([1, 1])))
    print(xor_perceptron_v1(np.array([1, 1])))

def main():
    test_add_perceptron()
    test_all_perceptron()

main()