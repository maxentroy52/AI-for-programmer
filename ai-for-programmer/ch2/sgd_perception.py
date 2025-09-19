#!/usr/bin/env python
# encoding: utf-8
import random

class SgdPerception:
    def __init__(self, eta, iterations, w, b):
        self.w = w
        self.b = b
        self.lr = eta
        self.iterations = iterations

    def fit(self, X, Y):
        for i in range(self.iterations):
            self.sgd_update_weights(X, Y)
            mse = self.cost_function(X, Y)
            print('{}th iteration, loss = {}'.format(i, mse))

    def predict(self, x):
        x = (x + 100) / 200  # Normalization
        return self.network_input(x)

    def update_weights(self, X, Y):
        batch_size = len(X)
        dw = 0.0
        db = 0.0
        for i in range(batch_size):
            # Calculate error based on actual network output (w*x + b)
            error = self.network_input(X[i]) - Y[i]
            dw += error * X[i]
            db += error
        dw = dw / batch_size * 2
        db = db / batch_size * 2

        self.w -= self.lr * dw
        self.b -= self.lr * db

    def sgd_update_weights(self, X, Y):
        dw = 0.0
        db = 0.0
        n = len(X)

        indexes = list(range(n))
        random.shuffle(indexes)
        batch_size = 4

        for k in range(batch_size):
            i = indexes[k]
            error = self.network_input(X[i]) - Y[i]
            dw += error * X[i]
            db += error
        dw = dw / batch_size * 2
        db = db / batch_size * 2

        self.w -= self.lr * dw
        self.b -= self.lr * db

    def network_input(self, x):
        return self.w * x + self.b

    def cost_function(self, X, Y):
        n = len(X)
        mse = 0
        for i in range(n):
            error = Y[i] - self.network_input(X[i])
            mse += (error * error)
        mse /= n
        return mse

x = [1, 2, 3, 10, 20, -2, -10, -100, -5, -20]
y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Create the NN with a slightly higher learning rate
model = SgdPerception(0.1, 500, 0, 0)  # Changed learning rate from 0.01 to 0.1

# Train the NN.
X = [(k + 100)/200 for k in x]
model.fit(X, y)

# Predict.
test_x = [30, 40, -20, -60]
for i in range(0, len(test_x)):
    test_y = model.predict(test_x[i])
    print('input {} => predict: {}'.format(test_x[i], test_y))

print("Final weights:")
print(model.w)
print(model.b)