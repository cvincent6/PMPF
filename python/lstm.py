## Colin Vincent
## Machine Learning
## Final Project
## LSTM For Price Prediction
## 12/11/17

## Adapted from: http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
## 			     http://vict0rsch.github.io/tutorials/keras/recurrent/#recurrent_keras_power.py

import os
import time
import numpy as np 
import pandas as pd 
import csv

from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import matplotlib.pyplot as plt

## ------------ Variables ------------

print "Running LSTM!"

start_time = time.time()
epochs = 1
seq_length = 50

## ------------ Data ------------

print "Reading Data..."

## Read in the dataset
dataset = pd.read_csv('dataset/training.csv')

## Pull out the price data
price = dataset['price']

## Convert to numpy array
data = price.values
print "# of Price values: " + str(data.size)

result = normalise_windows(result)

## Plot price data vs time
plt.plot(data)
plt.show()

train_start = 0
train_end = data.size * .9

test_start = train_end
test_end = data.size

train = data[: train_end]
test = data[train_end : test_end]

## ------------ Model ------------

print "Building Model..."

model = Sequential()

model.add(LSTM(
    input_dim=layers[0],
    output_dim=layers[1],
    return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(
    layers[2],
    return_sequences=False))

model.add(Dropout(0.2))

model.add(Dense(output_dim=layers[3]))

model.add(Activation("linear"))

start = time.time()

model.compile(loss="mse", optimizer="rmsprop")

print("Time : ", time.time() - start)

## Fit to data
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=epochs,
    validation_split=0.05)

## ------------ Prediction ------------

curr_frame = data[0]
predicted = []

for i in range(len(data)):
    predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
    curr_frame = curr_frame[1:]
    curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)


## ------------ Helper Functons ------------

def normalise_windows(window_data):
normalised_data = []
for window in window_data:
    normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
    normalised_data.append(normalised_window)
return normalised_data