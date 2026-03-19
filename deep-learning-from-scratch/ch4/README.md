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
- 但是，输入parameter是一个矩阵，numerical_gradient无法处理，只能交给numerical_gradient无法处理_nd处理
- 但是，本质当前的矩阵，也只是一个point or position.
- 同时，loss function里面会把parameter展开，展开后，数值微分的办法求导，就是给每一个parameter分量一个h，然后按照公式计算即可。