# Stateless
# data member
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

# test_apple_price()

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    # dout is the derivative with respect to the output variable.
    # dx and dy  are the derivatives with respect to the local variables.
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

def test_apple_and_orange_price():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_or_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    apple_orange_price = add_apple_or_orange_layer.forward(apple_price, orange_price)
    tax_price = mul_tax_layer.forward(tax, apple_orange_price)
    print(f"apple price: {apple_price}, orange price: {orange_price}")
    print(f"apple orange price: {apple_orange_price}, tax: {tax_price}")

    dtax_price = 1
    dtax, dapple_orange_price = mul_tax_layer.backward(dtax_price)
    dapple_price, dorange_price = add_apple_or_orange_layer.backward(dapple_orange_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(f"dtax: {dtax}, dapple_orange_price: {dapple_orange_price}")
    print(f"dapple_price: {dapple_price}, dorange_price: {dorange_price}")
    print(f"dapple: {dapple}, dapple_num: {dapple_num}")
    print(f"dorange: {dorange}, dorange_num: {dorange_num}")

test_apple_and_orange_price()