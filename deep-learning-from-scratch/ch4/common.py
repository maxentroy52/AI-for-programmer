import numpy as np

## 这里这么理解
## loss算的是每一个样本的平均loss，所以最后需要除以batch_size
## 前面的计算过程不变是因为python具备type flexibility，可以同时处理scalar and vector
## 否则，前面的计算还得再套一个np.sum
## eg: batch_size = 5, 5个样本，5个label [2, 7, 0, 9, 4](不是one-hot)
## 每个label去对应的样本里面，把对应的预测概率取出来，就是结果
## y[0, 2]
## y[1, 7]
## y[2, 0]
## y[3, 9]
## y[4, 4]
## [y[0, 2], y[1, 7], y[2, 0], y[3, 9], y[4, 4]] 这是一个数组啊
## np.log( [y[0, 2], y[1, 7], y[2, 0], y[3, 9], y[4, 4]] + 1e-7 )
##
## 所以，从这个输出中可以倒推出来，y和t的结构。
## t不是one-hot encoding，它就是告诉你index.
## y的输出，就是概率化分类的输出。它是一个向量，所以用t的index去取
## 当然，t如果是one-hot encoding，也是ok的。这么的好处就是可以直接矩阵乘法

"""
# Batch of 3 samples
y = np.array([
    [0.7, 0.2, 0.1],  # Sample 1
    [0.1, 0.8, 0.1],  # Sample 2
    [0.2, 0.3, 0.5]   # Sample 3
])
t = np.array([0, 1, 2])  # True labels for each sample

loss = cross_entropy_error(y, t)
print(f"Average Loss: {loss:.4f}")
# Calculation:
# Sample 1: -log(0.7) = 0.3567
# Sample 2: -log(0.8) = 0.2231
# Sample 3: -log(0.5) = 0.6931
# Average: (0.3567 + 0.2231 + 0.6931) / 3 = 0.4243
# Output: Average Loss: 0.4243
"""

"""
Why Reshape is Needed
The function needs to work in two scenarios:

Single sample: When you pass one data point
Batch of samples: When you pass multiple data points

# Test with single sample
y_single = np.array([0.7, 0.2, 0.1])
t_single = np.array([0])

# Output:
# === Testing Single Sample ===
# Original shapes - y: (3,), t: (1,)
# Detected single sample, reshaping...
# After reshape - y: (1, 3), t: (1, 1)   -->> [[0.7, 0.2, 0.1]] (1*3) 第一维只有一个sample, 第二维每个sample有3个元素 
# Batch size: 1
# Selected probabilities: [0.7]
# Final loss: 0.3567
"""

"""
y[np.arange(batch_size), t]
## y[0, 2]
## y[1, 7]
## y[2, 0]
## y[3, 9]
## y[4, 4]

这个展开是这样，语法非常简约，这里实现了遍历。
"""

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum( np.log(y[np.arange(batch_size), t] + delta) ) / batch_size

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x_max = np.max(x)
    x_new = x - x_max

    x_new_exp = np.exp(x_new)
    x_new_exp_sum = np.sum(x_new_exp)
    return x_new_exp / x_new_exp_sum

# 这里如果带入，那么w其实就是x
# (w,x,t)，x,t定了之后，loss是关于w的参数
# w有当前的数值，相当于loss(w)的一个点而已 要继续更新
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(len(x)):
        old_val = x[idx]

        # calculate f(x + h)
        x[idx] = old_val + h
        f1 = f(x)

        # calculate f(x - h)
        x[idx] = old_val - h
        f2 = f(x)

        # calculate partial derivative
        grad[idx] = (f1 - f2) / (2 * h)

        # restore the original value
        x[idx] = old_val

    return grad


def numerical_gradient_multi_array(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad

def numerical_gradient_nd(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num = 100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

# Function that computes sum of squares
def sum_of_squares(x):
    return np.sum(x ** 2)

def test_numerical_gradient():
    #print(numerical_gradient(sum_of_squares, np.array([[3.0, 4.0], [3.0, 2.0]])))
    #print(numerical_gradient(sum_of_squares, np.array([0.0, 2.0])))
    #print(numerical_gradient(sum_of_squares, np.array([3.0, 0.0])))

test_numerical_gradient()