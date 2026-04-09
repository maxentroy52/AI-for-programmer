sys.path.append(os.pardir)
from dataset.mnist import load_mnist

import numpy as np
from collections import OrderedDict
from common_layers import *
from common import *

class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Initialize weights.
        self.params = {}

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # Create layers(functional units).
        # 这里相当于把计算节点也抽象出来了
        # 对比my_training.py的实现 forward过程在predict中显示编码实现
        # 这里把forward过程抽象到layers当中
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        # The SoftmaxWithLoss layer combines softmax + loss calculation and is only used during training。
        # Same result - For classification, the predicted class is identical
        # 所以，这里单独放了一层。infer的时候不需要，只有training才需要。
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # For inference in the network, the output of the final affine layer is used in the inference result.
        # The unnormalized output result from a network is sometimes called a score.
        # To obtain only one answer in neural network inference, you only need to calculate the maximum score.
        # So, you do not need a softmax layer.
        # However, you do need a softmax layer in neural network training.
        for layer in self.layers.values():
            x = layer.forward(x)

        return x