## Loss function.


## Gradient method.

### Scalar

一元代码好实现
- 每一个元素是一个scalar
- 然后按照公式计算differences
- 此时，单个元素的微分是一个scalar

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

```

### Vector

多元实现
- 问题在于，每一个元素是一个vector,不是一个scalar.
- 按照公式计算的时候，对于每一个元素的每一个分量，需要单独计算partial differences.
- 此时，该元素的微分是一个vector，也是gradient.(collectively partial differences.)

下面是numeric graident求解多元函数的办法

$$ f(x_{1}, x_{2}) = x_{1}^{2} + x_{2}^{2} $$

$$ \frac{\partial f(x_{1}, x_{2})}{\partial x_{1}}  = \frac{ f(x_{1} + h, x_{2}) - f(x_{1} - h, x_{2}) }{2 * h}  $$

$$ \frac{\partial f(x_{1}, x_{2})}{\partial x_{2}}  = \frac{ f(x_{1}, x_{2} + h) - f(x_{1}, x_{2} - h) }{2 * h}  $$

$$ gradient(f(x_{1}, x_{2})) = (\frac{\partial f(x_{1}, x_{2})}{\partial x_{1}},\frac{\partial f(x_{1}, x_{2})}{\partial x_{2}}) $$

```python
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
```