## Loss function.


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