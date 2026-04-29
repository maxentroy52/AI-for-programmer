import numpy as np
import sys,os
import time

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

from common import plot_results
from two_layer_net import TwoLayerNet


def training():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

    # Hyper-parameters
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    training_loss_list = []

    iter_per_epoch = max(int(train_size / batch_size), 1)
    train_acc_list = []
    test_acc_list = []

    network = TwoLayerNet(784, 50, 10)

    for i in range(iters_num):
        start_time = time.time()

        # Step1: Obtain a mini-batch
        batch_mask = np.random.choice(train_size, batch_size, replace=False)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # Step2: Calculate a gradient
        grads = network.gradient(x_batch, t_batch)

        # Step3: Update the parameters
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grads[key]

        # Record end time
        end_time = time.time()
        elapsed_time = end_time - start_time  # Calculate elapsed time


        # Record the learning process
        loss = network.loss(x_batch, t_batch)
        training_loss_list.append(loss)
        print(f"iteration:{i} end, loss is {loss}, elapsed time is {elapsed_time:.2f}s")

        # Calculate recognition accuracy for each epoch
        # It should be printed in every epoch.
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"train_acc:{train_acc}, test_acc:{test_acc}")

    plot_results(training_loss_list, train_acc_list, test_acc_list)

if __name__ == '__main__':
    training()