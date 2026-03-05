import sys,os
sys.path.append(os.pardir)
import numpy as np
from common import*

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # init with a gaussian distribution

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss