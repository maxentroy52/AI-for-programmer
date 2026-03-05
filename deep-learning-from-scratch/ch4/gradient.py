import numpy as np
import matplotlib.pylab as plt

# Bad implementation sample
def numerical_differentiation_wrong(f, x):
    h = 1e-50                       ## 1e - 50, this is wrong.
    return (f(x + h) - f(x)) / h

def numerical_diff(f, x):
    h = 1e-4 # 0.001 1 * 10^(-4)
    return (f(x + h) - f(x - h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20, 0.1)
print(x)

y = function_1(x)
print(y)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

##############################################################

# now, x is a vector instead of a scalar.
def function_2(x):
    return np.sum(x**2)

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
        grad[idx] = (f1 - f2)/ (h*2)

        # restore the original value
        x[idx] = old_val

    return grad

grad1 = numerical_gradient(function_2, np.array([3.0, 4.0]))
print(grad1)

grad2 = numerical_gradient(function_2, np.array([0.0, 2.0]))
print(grad2)

grad3 = numerical_gradient(function_2, np.array([3.0, 0.0]))
print(grad3)

##############################################################

def gradient_descent(f, init_x, lr=0.01, step_num = 100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

init_x = np.array([-3.0, 4.0])
final_x = gradient_descent(function_2, init_x, lr = 0.1)
print(final_x)

final_x1 = gradient_descent(function_2, init_x, lr = 10)
print(final_x1)

final_x2 = gradient_descent(function_2, init_x, lr = 1e-10)
print(final_x2)

##################################################
### Gradients for a neural network
##################################################