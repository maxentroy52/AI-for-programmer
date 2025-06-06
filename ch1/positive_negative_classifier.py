import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## Create the NN.(Specify the structure of NN)
model = Sequential()
model.add(Dense(units=8, activation='relu', input_dim=1))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='sgd')

## Train the NN.
x = [1, 2, 3, 10, 20, -2, -10, -100, -5, -20]
y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
model.fit(x, y, epochs=1000, batch_size=4)

## Predict.
test_x = [30, 40, -20, -60]
test_y = model.predict(test_x)

## Print.
for i in range(0, len(test_x)):
    print('input {} => predict: {}'.format(test_x[i], test_y[i]))