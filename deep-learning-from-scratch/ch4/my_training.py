import numpy as np
from my_common import softmax
from my_common import sigmoid
from my_common import cross_entropy_error

class ImageRecognizerNN():
    ## ---------------------------- Basic method for ImageRecognizer-----------------------------------
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, weight_init_std = 0.01):
        # Model parameters.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = {}
        self.model['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.model['b1'] = np.zeros(hidden_size)
        self.model['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.model['b2'] = np.zeros(output_size)

        # Hyper parameters.
        self.learning_rate = learning_rate

    def debug(self):
        print(self.model['W1'].shape)
        print(self.model['b1'].shape)
        print(self.model['W2'].shape)
        print(self.model['b2'].shape)

    ## ---------------------------- Forward process for ImageRecognizer-----------------------------------
    def predict(self, x):
        a1 = np.dot(x, self.model['W1']) + self.model['b1']
        z1 = sigmoid(a1)

        a2 = np.dot(z1, self.model['W2']) + self.model['b2']
        y = softmax(a2)
        return y

    ## ---------------------------- Training process for ImageRecognizer-----------------------------------
    ## Training process for ImageRecognizer
    # First, we need to define the loss function.
    # Raw sample + label = x + t
    # In offline process, we need to perform the prediction again.
    # Which means, another forward propagation is performed.
    def loss(self, x, t):
        # perform forward propagation.
        y = self.predict(x)

        # y is the predicted one, t is the true one.
        # Now, we have both of them, we can calculate the loss.
        #
        # There is another important thing you should notice:
        # x and t can be multiple samples, so the cross_entropy_error
        # performs the average loss based on these samples.
        loss = cross_entropy_error(y, t)
        return loss

def test_image_recognizer():


    image_recognizer = ImageRecognizerNN(784, 100, 10)
    image_recognizer.debug()

test_image_recognizer()