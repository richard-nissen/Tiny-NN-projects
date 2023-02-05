from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

# Create a sequential model
model = Sequential()

# Add a dense layer with units = 4, input_dim = 3, and activation = sigmoid
model.add(Dense(units=4, input_dim=3, activation='sigmoid'))

# Add a dense layer for output with units = 1 and activation = sigmoid
model.add(Dense(units=1, activation='sigmoid'))

# Print the model summary
model.summary()

# Create a stochastic gradient descent optimizer with learning rate = 1
sgd = SGD(lr=1)

model.compile(loss='mean_squared_error', optimizer=sgd)

# Specify random seed
np.random.seed(42)

X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

y = np.array([[0],[1],[1],[0]])

# Train the model for 1500 epochs
model.fit(X, y, epochs=1500, verbose=0)

# Print the model weights
print(model.get_weights())

# Predict the output for X
print(model.predict(X))

