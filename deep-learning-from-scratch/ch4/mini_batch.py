import sys,os
import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def show():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize=True, one_hot_label=True)
    print(x_train.shape)
    print(x_test.shape)

    print(t_train.shape)
    print(t_test.shape)

    train_size = x_train.shape[0]
    batch_size = 10

    print(train_size, batch_size)

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    print(batch_mask)
    print(x_batch.shape)
    print(t_batch.shape)

show()