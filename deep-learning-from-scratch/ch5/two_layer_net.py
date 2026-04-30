import numpy as np
from collections import OrderedDict
from common_layers import *
from common import *

class TwoLayerNet():
    ## ---------------------------- Basic method for TwoLayerNet -----------------------------------
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

    ## ---------------------------- Forward process for TwoLayerNet -----------------------------------
    def predict(self, x):
        # For inference in the network, the output of the final affine layer is used in the inference result.
        # The unnormalized output result from a network is sometimes called a score.
        # To obtain only one answer in neural network inference, you only need to calculate the maximum score.
        # So, you do not need a softmax layer.
        # However, you do need a softmax layer in neural network training.
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def accuracy(self, x, t):
        """
        Step 1 - Raw predictions (probabilities):
        Shape: (10, 3)
        y_pred =
         [[0.8 0.1 0.1]
         [0.2 0.7 0.1]
         [0.1 0.2 0.7]
         [0.6 0.3 0.1]
         [0.3 0.6 0.1]
         [0.1 0.1 0.8]
         [0.9 0.05 0.05]
         [0.2 0.2 0.6]
         [0.1 0.8 0.1]
         [0.7 0.2 0.1]]

        Step 2 - Predicted class indices (argmax on axis=1):
        Shape: (10, 1)
        y_pred = [0 1 2 0 1 2 0 2 1 0]
        (0=Cat, 1=Dog, 2=Bird)

        Step 3 - True class indices (argmax on axis=1):
        # CREATE TRUE LABELS (one-hot encoded) for 10 samples
        t_test = np.array([
            [1, 0, 0],  # Sample 1: Actually a Cat (class 0)
            [0, 1, 0],  # Sample 2: Actually a Dog (class 1)
            [0, 0, 1],  # Sample 3: Actually a Bird (class 2)
            [1, 0, 0],  # Sample 4: Actually a Cat (class 0)
            [0, 1, 0],  # Sample 5: Actually a Dog (class 1)
            [0, 0, 1],  # Sample 6: Actually a Bird (class 2)
            [1, 0, 0],  # Sample 7: Actually a Cat (class 0)
            [0, 0, 1],  # Sample 8: Actually a Bird (class 2)
            [0, 1, 0],  # Sample 9: Actually a Dog (class 1)
            [1, 0, 0]   # Sample 10: Actually a Cat (class 0)
        ])
        t_test = [0 1 2 0 1 2 0 2 1 0]
        (0=Cat, 1=Dog, 2=Bird)

        Step 4 - Which predictions are correct?
        y_pred == t_test = [ True  True  True  True  True  True  True  True  True  True]
        Number correct = 10

        Step 5 - Calculate accuracy:
        Accuracy = 10 / 10.0 = 1.0
        """

        # 每一个y是多维向量，每个类别的概率
        # 每一个t是多维向量，one-hot encoding
        # 所以，他们两可以直接相乘
        # 这里，拍平了是为了算准确率，也是没问题的，要拍平，都拍平。
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    ## ---------------------------- Backward process for TwoLayerNet -----------------------------------
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)


    def gradient_numerical(self, x, t):
        grads = {}

        loss_W = lambda W: self.loss(x, t)

        grads['W1'] = numerical_gradient_nd(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_nd(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_nd(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_nd(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # settings
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads