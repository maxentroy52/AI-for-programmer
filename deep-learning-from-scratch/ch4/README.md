## Gradient method.

梯度是这样，只有当我们讨论多元函数的时候，才会有梯度的概念。他是偏导数的集合。

下面是numeric graident求解多元函数的办法

$$ f(x_{1}, x_{2}) = x_{1}^{2} + x_{2}^{2} $$

$$ \frac{\partial f(x_{1}, x_{2})}{\partial x_{1}}  = \frac{ f(x_{1} + h, x_{2}) - f(x_{1} - h, x_{2}) }{2 * h}  $$

$$ \frac{\partial f(x_{1}, x_{2})}{\partial x_{2}}  = \frac{ f(x_{1}, x_{2} + h) - f(x_{1}, x_{2} - h) }{2 * h}  $$

$$ gradient(f(x_{1}, x_{2})) = (\frac{\partial f(x_{1}, x_{2})}{\partial x_{1}},\frac{\partial f(x_{1}, x_{2})}{\partial x_{2}}) $$

- 简单总结下：
- 对于每一个分量，给它一个增量，然后代入公式求解
- 所以，代码实现就是遍历所以分量，执行上述操作即可。
- 注意，X是一个position，每一个element是一个分量。

```python
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

# Function that computes sum of squares
def sum_of_squares(x):
    return np.sum(x ** 2)

print(numerical_gradient(sum_of_squares, np.array([3.0, 4.0]))) # calculate the gradient of point(3, 4)
print(numerical_gradient(sum_of_squares, np.array([0.0, 2.0]))) # calculate the gradient of point(0, 2)
print(numerical_gradient(sum_of_squares, np.array([3.0, 0.0]))) # calculate the gradient of point(3, 0)

[6. 8.]
[0. 4.]
[6. 0.]
```

梯度的物理意义，这里展开一下。其实和导数是一回事。
- ```numerical_gradient(sum_of_squares, np.array([3.0, 4.0]))``` 它的物理意义是，sum_of_squares这个函数，在(3,4)这个点，沿着(6,8)这个方向，可以达到函数的可能极大值。
- 下面两个同理，sum_of_squares这个函数在(0,2)这个点，沿着(0, 4)这个方向，可以拿到极大值。
- 在(3, 0)这个点，沿着(6, 0)这个方向，可以拿到极大值。
- 这个方向，只是个合力的方向。如果不好理解，就拆成偏导数理解，每一个变量自身，无非就是正方向和负方向两个方向而已。
- 导数的方向，正方向，按照这个方向，可以获得极大。
- 梯度的方向，无非是两个导数方向的合力方向

>To be more precise, the direction of a gradient is the direction that increases the value
of the function most at each position.

补充一点，```def numerical_gradient(f, x):```，这个函数的设计，需要理解。
- 定义写成f(x)没有问题，其中x是自变量。求关于它的导数。
- 微分法的办法是在当前position给一个h，然后求解。注意，此时position是确定的，已经给出了。
- 所以，x的具体体现是一个已经给出值的数组。
- 那么还需要给出x的意义在哪里呢？f(theta; x, t)，这表明f是关于theta的函数，所以调用的时候，需要给出f, theta_array
  - 这样，在theta的position进行近似微分
  - 如果给出t的postion，那么则是在t的postion进行微分。
  - 所以，x的作用，是确定f到底是谁的函数。是谁的都行，所以需要参数给出确定。

### Vector

- 这里需要特别注意的是，上面的函数，是个多元函数，处理的x是一个point or postion.
- 但是，这每一个元，还可能是个vector, more than a scalar

比如说，我有一个函数，需要判断学生是否优秀。y = 0.1 * height + 0.2 * weight + 0.7 * grades.
(height, weight, grades)是一个point or position没问题，但他们都是scalar.

那么我们再看一个计算方式 y = 0.4 * non_final_grades(math, english, chinese) + 0.6 * final_grades(math, english, chinese)
[(88, 89, 91), (89,88,88)] 所以，这是一个点的坐标。再理解上要注意。

当然，这里不好理解怎么求？其实，不管输入怎么变化，1d, 2d, nd，都只是一个点而已。function最终会把他们拍平的。

比如 y = 0.4 * x1 + 0.4 * x2 + 0.4 * x3 + 0.6 * x4 + 0.6 * x5 + 0.6 * x6

- 拿我们的例子来说，模型参数更新，本质是loss function计算当前parameter的gradient，然后用gradient method更新parameter
- 但是，输入parameter是一个矩阵，numerical_gradient无法处理，只能交给numerical_gradient_nd处理
- 但是，本质当前的矩阵，也只是一个point or position.
- 同时，loss function里面会把parameter展开，展开后，数值微分的办法求导，就是给每一个parameter分量一个h，然后按照公式计算即可。

所以，由于loss function的计算对于parameter本质是拍平的。所以，求导过程也就是一次迭代。只不过迭代的实现，跟之前for-range-loop
不太一样。

```python
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
```

### Example

这里结合我们具体的例子说一下：

```python
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient_nd(loss_w, self.params['W1'])
        grads['W2'] = numerical_gradient_nd(loss_w, self.params['W2'])
        grads['b1'] = numerical_gradient_nd(loss_w, self.params['b1'])
        grads['b2'] = numerical_gradient_nd(loss_w, self.params['b2'])
```

- 上文我们强调，这是一个point or position。但是，我们知道，loss的parameter是一个整体，也即point or position，为什么分开。
- 首先，在计算上来说，放到一起，和分开放没有关系。比如第一个式子，我给W1的每一个分量一个增量h,W2/b1/b2都是不变的。所以，计算上是OK的。
- 分开放的原因，原来的parameter就是分开放的。也可以放成一个point(w11, w12, w21, w22, b11, b12, b21, b22)
- 看下面代码
  - self.params分别存储了几组参数
  - 计算梯度也要分别计算他们的梯度
  - 然后分别更新

```python
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        grads = {}
        grads['W1'] = numerical_gradient_nd(loss_w, self.params['W1'])
        grads['W2'] = numerical_gradient_nd(loss_w, self.params['W2'])
        grads['b1'] = numerical_gradient_nd(loss_w, self.params['b1'])
        grads['b2'] = numerical_gradient_nd(loss_w, self.params['b2'])
```

那么我们看最后参数的更新：
- 拿到当前参数
- 计算当前参数的梯度
- 然后用梯度下降，更新当前参数，拿到新参数
```python
def gradient_descent(f, init_x, lr=0.01, step_num = 100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
```

## Training.

说一下training，这个还有点东西。

```python
# Hyper-parameters
iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

for i in range(iter_num):
    # Obtain a mini-batch
    # 这里 每次拿到的是100个下标 就是100个sample
    batch_mask = np.random.choice(train_size, batch_size)
    #print(batch_mask)

    # 不得不说 非常简练抽象的表达
    # 本次的训练样本我拿到 下面就开始训练
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 训练 - 美其名曰 本质就是计算loss function对于
    # 当前参数的梯度
    # Calculate a gradient
    # 它这个逻辑是这样 loss = f(x, t, theta)
    # x, t作为常数带入后
    # loss变成 loss = f(theta; x,t)
    grad = network.numerical_gradient(x_batch, t_batch)

    # Update the parameter.
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Record learning process
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
```

### Policy(少量batch多次迭代)

说一下训练策略，可以发现，采用了如下办法
- 没有total batch, 而是mini batch
- 没有只训练一次，而是迭代了多次。每次迭代，更新一次参数。

1. 更频繁的参数更新 → 更快收敛 
- 参数更新次数：全批量只更新1次参数，小批量更新10,000次
- 虽然总计算量更大，但每次更新后的参数都在改善，后续的batch是在"更好"的参数上计算

2. 梯度噪声 → 更好的泛化能力
- 这是随机性带来的好处
- 每次选择随机样本，进行带噪声的梯度估计。
- 噪声可以帮助跳出局部最小，找到更好的全局最优。

3. 内存效率 → 不需要一次性加载所有数据

- 类比理解
想象你要调整100个旋钮让机器达到最优状态：
- 全批量：测试所有60,000种情况，计算平均表现，然后调整一次旋钮。重复这个慢过程。
- 小批量：随机选100种情况测试，立即微调旋钮。虽然测试总数更多，但每次微调后机器就在改进，整体更快找到最优。

### Optimization

目前实际的做法，也就是优化后的做法。

```python
# 损失函数对batch的梯度 = 各个样本梯度的平均
# ∇L_batch = (1/100) * Σ ∇L(sample_i)

# 关键：计算每个∇L(sample_i)时，使用的是相同的参数θ
# 这些计算是独立的！

import numpy as np
import time

# 模拟100个样本，每个784维
samples = np.random.randn(100, 784)
weights = np.random.randn(784, 100)

# 方式1：循环（串行）
start = time.time()
for i in range(100):
    result = np.dot(samples[i], weights)  # 100次单独计算
print(f"循环耗时: {time.time() - start:.4f}秒")

# 方式2：向量化（并行）
start = time.time()
result = np.dot(samples, weights)  # 1次矩阵运算！
print(f"向量化耗时: {time.time() - start:.4f}秒")

# 输出示例：
# 循环耗时: 0.0234秒
# 向量化耗时: 0.0003秒  # 快约78倍！
```

### Lambda的使用

Your code flow:

- numerical_gradient_nd(loss_W, self.model['W1']) is called
- Inside numerical_gradient_nd, x points to self.model['W1']
- x[idx] = tmp_val + h directly modifies self.model['W1'][idx]
- fxh1 = f(x) calls loss_W, which calls self.loss(x, t)
- self.loss uses self.model['W1'] (which now has the perturbed value)
- After calculation, x[idx] = tmp_val restores the original value

AI mistakenly thought the lambda was ignoring the w parameter, but I missed that:
- The w parameter is not actually used in the lambda body
- The real "perturbation" happens by directly modifying the array before calling f(x)
- The lambda simply calls self.loss(x, t) which reads the current values from the model parameters

### Summary

- 一个迭代，更新一次。
- 每个迭代，在上个迭代基础上，参数计算并更新。

```
迭代1 (iter=1)         迭代2 (iter=2)        迭代3 (iter=3)
    ↓                     ↓                     ↓
[Batch 1]              [Batch 2]              [Batch 3]
    ↓                     ↓                     ↓
[并行处理100个样本]  [并行处理100个样本]  [并行处理100个样本]
    ↓                     ↓                     ↓
 更新参数θ₁           更新参数θ₂           更新参数θ₃
    ↓                     ↓                     ↓
   必须串行！          必须串行！          必须串行！
```

## Analysis

分析一下训练的结果：
- 从日志中可以看出来，非常慢。(loss在不断下降，训练是有效的)
- 1min可以进行3次迭代，1h可以进行180次迭代，1day可以进行4320次迭代
- 10000次迭代需要2天多。太慢了，下面分析一下慢在哪。
```python
iteration:0 begins.
iteration:0 end, loss is 6.903138495471898, elapsed time is 19.42s
iteration:1 begins.
iteration:1 end, loss is 6.8894490408905495, elapsed time is 19.66s
iteration:2 begins.
iteration:2 end, loss is 6.893976967606961, elapsed time is 19.65s
iteration:3 begins.
iteration:3 end, loss is 6.889668194742657, elapsed time is 19.83s
iteration:4 begins.
iteration:4 end, loss is 6.8949818734213935, elapsed time is 19.92s
iteration:5 begins.
iteration:5 end, loss is 6.874609642977018, elapsed time is 20.24s
iteration:6 begins.
iteration:6 end, loss is 6.90435747443245, elapsed time is 20.35s
iteration:7 begins.
iteration:7 end, loss is 6.879768286747026, elapsed time is 20.78s
...
iteration:2392 end, loss is 5.140628341661429, elapsed time is 21.98s
iteration:2393 begins.
iteration:2393 end, loss is 5.124200654084805, elapsed time is 22.21s
iteration:2394 begins.
iteration:2394 end, loss is 5.2101359276571575, elapsed time is 22.48s
```

### For your network

- Hidden layer: 784 × 100 = 78,400 weights + 100 biases = 78,500 parameters
- Output layer: 100 × 10 = 1,000 weights + 10 biases = 1,010 parameters
- Total parameters: ~79,510 parameters

### Computation per iteration:
- Each parameter requires 2 forward passes (f(x+h) and f(x-h))
- Total forward passes per iteration: 79,510 × 2 = 159,020 forward passes
- Each forward pass computes ~100 × (784×100 + 100×10) ≈ 7.94 million operations
- Total operations per iteration: ~1.26 trillion floating-point operations!

上面这个分析，我们拆解一下，核心思路是这样
- 首先，先计算出来，需要多少次FP
  - 每个参数，两次FP
  - 总共79510个参数
  - 总计需要79510 * 2 = 159020次FP
- 然后，计算出FP的开销
  - 这个就是矩阵乘法的开销
  - 我们先考虑batch size为1的情形。
  - 输入x是784维度的向量，然后做矩阵乘法(1*784) * (784 * 100)，总共784 * 100次计算
    - 行向量乘以100个列向量
    - 每个行向量是784维，所以784 * 1次计算
    - 总共就是784 * 100
  - output layer同理，188 * 10
  - 总共就是784 * 100 + 100 * 10
  - 100个样本，就是100*(784 * 100 + 100 * 10) = 7940000次计算
- 总共 159020 * 7940000 = 1.26 trillion floating-point operations!