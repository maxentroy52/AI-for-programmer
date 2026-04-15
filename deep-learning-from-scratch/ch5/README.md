## Demo

说一下backward函数写法的一些技巧
- backward的过程和forward正好相反
- 所以，参数也是反的。
  - forward的输出，output variable，是backward的input variable, 语义是derivative with output variables.
  - forward的输入，local variable，是backward的output variable，语义是derivative with input variables.

forward是复合函数由里向外的过程，backward则是复合函数由外向里的过程。
- 由里向外，其实就是函数的计算过程。
- 由外向里，是导数的计算过程。

```python
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    # On the other hand, backward multiplies the derivative from the upstream(dout)
    # by the reserved value of forward propagation and passes the result downstream.
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

def test_apple_price():
    apple = 100
    apple_num = 2
    tax = 1.1

    # Layer
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    print(f"apple price: {apple_price}, tax price: {price}")

    # backward
    # You can use backward() to obtain the differential of each variable.
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    print(f"dapple price: {dapple_price}, dtax: {dtax}")
    print(f"dapple: {dapple}, dapple_num: {dapple_num}")
```

## Affine

这个算子也简单说下，根据demo的讨论
- forward的输入是x，这个是backward的输出。这意味着backward需要对谁求导
- backward的输入自然是dout

1. 这里的疑问是，dw,db并不是forward输入，为什么要对他们求导？能问出这个问题，证明对DL的理解不到位。
2. 我们这里求导数，到底是在干什么？前面讲了太多求导技巧，以至于把最关键的问题搞混了。

我们求导的根本目的不还是通过SGD这种启发式方法，来更新参数吗？所以，dW,dB是一定需要求的！！！这个没有疑问。反过来，该问的是，为什么要求dx?
- y = x * w + b
- 还是从公式上来说，我们拍平了看，是不是y = x1 * x2 + x3，那么，如果需要反向传播，是不是需要对所有x1,x2,x3都求partial derivative.
- 所以，你看backward，就是把dx,dw,db全部求出来。dx直接返回了，dw,db都缓存了。

```python
class Affine:
    """
    Affine (Fully Connected) Layer.

    Performs linear transformation: out = x · W + b
    where:
    - x is input from previous layer (batch of samples)
    - W is weight matrix (learnable parameters)
    - b is bias vector (learnable parameters)

    This layer is the fundamental building block of neural networks,
    connecting all neurons between layers.
    """

    def __init__(self, W, b):
        """
        Initialize Affine layer with weights and bias.

        Args:
            W: Weight matrix of shape (input_dim, output_dim)
               Each column represents weights for one output neuron
            b: Bias vector of shape (output_dim,)
               Bias term added to each output neuron
        """
        # Learnable parameters
        self.W = W  # Weights connecting input to output
        self.b = b  # Bias terms

        # Cache variables for backward pass
        self.x = None  # Store input from forward pass (used to compute gradients)
        self.dW = None  # Gradient of loss with respect to weights
        self.db = None  # Gradient of loss with respect to bias

    def forward(self, x):
        """
        Forward pass: Compute output = x @ W + b

        Args:
            x: Input data of shape (batch_size, input_dim)
               Each row is one training sample

        Returns:
            out: Output of shape (batch_size, output_dim)
                 Each row is the transformed feature vector for one sample

        Mathematical formula:
            out_ij = Σ_k x_ik * W_kj + b_j
        """
        # Cache input for backward pass (needed to compute dW)
        self.x = x

        # Linear transformation: (batch_size, input_dim) @ (input_dim, output_dim)
        # Result shape: (batch_size, output_dim)
        # Add bias broadcasting: bias (output_dim,) is added to each row
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        """
        Backward pass: Compute gradients and propagate error backwards.

        Args:
            dout: Gradient of loss with respect to output
                  Shape: (batch_size, output_dim)

        Returns:
            dx: Gradient of loss with respect to input
                Shape: (batch_size, input_dim)
                Propagated to previous layer

        Gradients computed:
            dW: Gradient for weight update (batch_size, output_dim)
            db: Gradient for bias update (output_dim,)

        Mathematical derivations:
            Let L be loss function.
            Given: dout = ∂L/∂out

            ∂L/∂x = ∂L/∂out · ∂out/∂x = dout · W^T
            ∂L/∂W = x^T · dout
            ∂L/∂b = Σ(dout) over batch dimension (axis=0)
        """
        # Compute gradient with respect to input (for backpropagation to previous layer)
        # dx = dout * W^T
        # Shape: (batch_size, output_dim) @ (output_dim, input_dim) = (batch_size, input_dim)
        dx = np.dot(dout, self.W.T)

        # Compute gradient with respect to weights
        # dW = x^T * dout
        # Shape: (input_dim, batch_size) @ (batch_size, output_dim) = (input_dim, output_dim)
        self.dW = np.dot(self.x.T, dout)

        # Compute gradient with respect to bias
        # db = Σ(dout) over batch samples (axis=0)
        # Shape: (output_dim,)
        # We sum because bias is shared across all samples in the batch
        self.db = np.sum(dout, axis=0)
```

## 梯度计算

看下面代码，其实上一章已经讨论过了，这里我再讨论一下。

- loss_W当中，lambda函数体，并没有真的使用将lambda参数传入，看起来不需要。但内部是使用的，这里只是形式。
- loss_W = f(theta, x, t)，至于到底是谁的函数，其实取决于你希望它是谁的函数。
- 在numerical_gradient_nd的计算当中，我们其实可以看出俩。这个函数 ```numerical_gradient_nd(f, x)```
  - 自变量是x，意味着会对x进行取增量，然后计算微分
  - ```numerical_gradient_nd(loss_W, self.params['W1'])``` 从实际调用中，可以看出来，```x = self.params['W1']```
  - 所以，自变量是W1
  - 带入sample(x, t)也是重要的，这样loss_w的形式才能最终定下来。

```python
    def gradient_numerical(self, x, t):
        grads = {}

        loss_W = lambda W: self.loss(x, t)

        grads['W1'] = numerical_gradient_nd(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_nd(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_nd(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_nd(loss_W, self.params['b2'])

        return grads
```

以上，其实我之前一直忽略了一点，就是在training的时候，我们计算loss，**本质也是一次forward**