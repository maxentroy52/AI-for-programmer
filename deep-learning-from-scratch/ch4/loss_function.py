import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch


def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def test_sum_squared_error():
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

    print(sum_squared_error(np.array(y), np.array(t)))

    y1 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print(sum_squared_error(np.array(y1), np.array(t)))

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def test_cross_entropy_error():
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

    print(cross_entropy_error(np.array(y), np.array(t)))

    y1 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print(cross_entropy_error(np.array(y1), np.array(t)))

test_sum_squared_error()
test_cross_entropy_error()

"""
"if a single sample is given, we transform it to batch samples"
y = [0.1, 0.05, 0.6, 0.0, 0.05]  # One person's answers
t = [2]                           # One correct answer

y = [[0.1, 0.05, 0.6, 0.0, 0.05]]  # Now it's a "batch" of 1
t = [[2]]                           # Now it's a "batch" of 1

Because the rest of the function is designed for batches.
"""

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))