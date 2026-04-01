## Demo

说一下backward函数写法的一些技巧
- backward的过程和forward正好相反
- 所以，参数也是反的。
  - forward的输出，output variable，是backward的input variable, 语义是derivative with output variables.
  - for的输入，local variable，是backward的output variable，语义是derivative with input variables.

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