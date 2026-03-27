import numpy as np
import sys,os
import time

from my_common import softmax
from my_common import sigmoid
from my_common import cross_entropy_error
from my_common import numerical_gradient_nd
from my_common import plot_results

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

class ImageRecognizerNN():
    ## ---------------------------- Basic method for ImageRecognizer-----------------------------------
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, learning_rate=0.1, iterations=10000, batch_size=100):
        # Model parameters.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = {}
        self.model['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.model['b1'] = np.zeros(hidden_size)
        self.model['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.model['b2'] = np.zeros(output_size)

        # Hyperparameters for training.
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size

    def debug(self):
        print("---------------------Model's parameters:---------------------")
        print(self.model['W1'].shape)
        print(self.model['b1'].shape)
        print(self.model['W2'].shape)
        print(self.model['b2'].shape)
        print(f"learning rate is: {self.learning_rate}")
        print(f"iterations is: {self.iterations}")
        print(f"batch size is: {self.batch_size}")

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

    # calculate gradient for all parameters.
    def gradient_numerical(self, x, t):
        grads = {}

        loss_W = lambda W: self.loss(x, t)

        grads['W1'] = numerical_gradient_nd(loss_W, self.model['W1'])
        grads['b1'] = numerical_gradient_nd(loss_W, self.model['b1'])
        grads['W2'] = numerical_gradient_nd(loss_W, self.model['W2'])
        grads['b2'] = numerical_gradient_nd(loss_W, self.model['b2'])

        return grads

    def train(self, x_train, t_train, x_test, t_test):
        print("---------------------Training begins:---------------------")
        loss_list = []
        train_size = x_train.shape[0]
        iter_per_epoch = max(int(train_size / self.batch_size), 1)
        train_acc_list = []
        test_acc_list = []

        for iteration in range(self.iterations):
            start_time = time.time()
            print(f"iteration:{iteration} begins.")

            # Obtain a mini-batch
            # 这里 每次拿到的是100个下标 就是100个sample
            batch_mask = np.random.choice(train_size, self.batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # calculate the gradient for the model parameters
            grads = self.gradient_numerical(x_batch, t_batch)

            # update the model parameters based on mini-batch samples.
            self.model['W1'] -= self.learning_rate * grads['W1']
            self.model['b1'] -= self.learning_rate * grads['b1']
            self.model['W2'] -= self.learning_rate * grads['W2']
            self.model['b2'] -= self.learning_rate * grads['b2']

            # calculate the loss based on the new model
            loss = self.loss(x_batch, t_batch)
            loss_list.append(loss)

            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time

            print(f"iteration:{iteration} end, loss is {loss}, elapsed time is {elapsed_time:.2f}s")

            # Calculate recognition accuracy for each epoch
            # It should be printed in every epoch.
            if iteration % iter_per_epoch == 0:
                train_acc = self.accuracy(x_train, t_train)
                test_acc = self.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print(f"train_acc:{train_acc}, test_acc:{test_acc}")

        plot_results(loss_list, train_acc_list, test_acc_list)

    ## ---------------------------- Acc calculation -----------------------------------
    def accuracy(self, x_test, t_test):
        y_pred = self.predict(x_test)
        y_pred = y_pred.argmax(axis=1)
        t_test = t_test.argmax(axis=1)

        accuracy = np.sum(y_pred == t_test)
        return accuracy / float(x_test.shape[0])

def test_image_recognizer():
    image_recognizer = ImageRecognizerNN(784, 100, 10)
    image_recognizer.debug()

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

    print("---------------------Sample's parameters:---------------------")
    print(x_train.shape, t_train.shape)
    print(x_test.shape, t_test.shape)

    image_recognizer.train(x_train, t_train, x_test, t_test)

test_image_recognizer()