import sys, os
import numpy as np

sys.path.append(os.pardir)
from common import *

# step1 - mini-batch
# step2 - calculating gradients
# step3 - update parameters(gradients descent)
# step4 - repeating

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W1'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(output_size)

def predict(self, x):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)