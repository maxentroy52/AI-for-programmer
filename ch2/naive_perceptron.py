#!/usr/bin/env python
# encoding: utf-8

class NaivePerceptron:
    def __init__(self, eta = 0.01, iterations = 10):
        self.lr = eta
        self.iterations = iterations
        self.w = 0.0
        self.bias = 0.0

    def fit(self, X, Y):
        for _ in range(self.iterations):
            for i in range(len(X)):
                x = X[i]
                y = Y[i]

                ## Learning policy is naive.
                update = self.lr * (y - self.predict(x))
                self.w += update * x
                self.bias += update

    def network_input(self, x):
        return self.w * x + self.bias

    def predict(self, x):
        return 1.0 if self.network_input(x) > 0.0 else 0.0

x = [1, 2, 3, 10, 20, -2, -10, -100, -5, -20]
y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

## Create the NN.
model = NaivePerceptron(0.01, 10)

## Train the NN.
model.fit(x, y)

## Predict.
test_x = [30, 40, -20, -60]
for i in range(0, len(test_x)):
    test_y = model.predict(test_x[i])
    print('input {} => predict: {}'.format(test_x[i], test_y))

print(model.w)
print(model.bias)