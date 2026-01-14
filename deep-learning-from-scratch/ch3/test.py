import numpy as np
from activation_function import *

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

x = np.array([1010, 1000, 990])
y = softmax(x)
print(y)

