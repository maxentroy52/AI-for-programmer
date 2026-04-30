import sys,os
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:10]
t_batch = t_train[:10]

grads_numerical = network.gradient_numerical(x_batch, t_batch)
grads_backward = network.gradient(x_batch, t_batch)

for key in grads_numerical.keys():
    diff = np.average(np.abs(grads_numerical[key] - grads_backward[key]))
    print(key + ":" + str(diff))