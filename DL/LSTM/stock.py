import numpy as np
import pandas as pd  # to read data like p.read('data.csv') or else
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Creating dummy data - an array of 100 ones for illustration
data = np.array([[1] for i in range(100)])

# Preparing input and output sequences
# X will be all elements except the last one
X = data[:-1]  # input sequence
# y will be all elements except the first one
y = data[1:]   # output sequence

# Reshaping data for LSTM
# LSTM expects input shape: (samples, time steps, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Building a simple LSTM model
model = Sequential()
# Adding LSTM layer with 50 units, input shape of (1 timestep, 1 feature)
model.add(LSTM(50, input_shape=(1, 1)))
# Adding Dense layer with 1 unit for output
model.add(Dense(1))

# Compiling the model
# Using 'adam' optimizer and mean squared error loss function
model.compile(optimizer='adam', loss='mse')

# Training the model
# Fitting for 50 epochs with verbose=0 (silent training)
model.fit(X, y, epochs=50, verbose=0)

# Making predictions
predicted = model.predict(X)
print(predicted)