## Colin Vincent

import os
import time
import numpy as np 
import pandas as pd 
import csv
import warnings

from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

# Data

file = 'dataset/training.csv'

with open(file) as f:
        data = csv.reader(f, delimiter=",")
        power = []
        nb_of_values = 0
        for line in data:
            try:
                power.append(float(line[0]))
                nb_of_values += 1
            except ValueError:
                pass

result = []

for index in range(len(power) - 24):
    result.append(power[index: index + 24])

result = np.array(result)  # shape (2049230, 50)

result_mean = result.mean()
result -= result_mean
print "Shift : ", result_mean
print "Data  : ", result.shape

row = int(round(0.9 * result.shape[0]))
train = result[:row, :]
np.random.shuffle(train)

X_train = train[:, :-1]
y_train = train[:, -1]
X_test = result[row:, :-1]
y_test = result[row:, -1]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
layers = [1, 50, 100, 1]

model.add(LSTM(
    input_dim=layers[0],
    output_dim=layers[1],
    return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(
    layers[2],
    return_sequences=False))

model.add(Dropout(0.2))

model.add(Dense(
        output_dim=layers[3]))

model.add(Activation("linear"))

start = time.time()
model.compile(loss="mse", optimizer="rmsprop")

print "Compilation Time : ", time.time() - start

## ---------- Running NN

epochs = 100
ratio = 0.5

if model is None:
    model = build_model()

try:
	model.fit(
	    X_train, y_train,
	    batch_size=len(power)/24, nb_epoch=epochs, validation_split=0.05)
	predicted = model.predict(X_test)
	predicted = np.reshape(predicted, (predicted.size,))

except KeyboardInterrupt:
	print 'Training duration (s) : ', time.time() - global_start_time

print len(y_test)
print len(predicted)

try:
    fig = plt.figure()
    plt.plot(predicted[:24])
    plt.plot(y_test[:24])
    plt.legend()
    plt.show()
except Exception as e:
    print str(e)
