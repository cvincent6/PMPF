## Colin Vincent
## Machine Learning 
## LSTM for Power Market Price Prediction
## December 14, 2017

## Adapted from:
## http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction

import os
import time
import numpy as np 
import pandas as pd 
import csv
import warnings

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from numpy import newaxis

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

## Hiding Warnings from Tensorfow etc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

## Global Variables
TIME_LENGTH = 24
EPOCHS = 50
RATIO = 0.5
LAYERS = [1, 50, 100, 1]
VALIDATION_SPLIT = 0.05

## ----------- Reading in Dataset -----------

file_name = 'dataset/training.csv'

with open(file_name) as file:
        dataset = csv.reader(file, delimiter=",")

        # Dataset has power and load info
        power = []
        load = []

        num_values = 0

        for line in dataset:
            try:
                power.append(float(line[0]))
                load.append(float(line[1]))
                num_values += 1
            except ValueError:
                print "Cound't pull data at index: " + str()

data = []

## Plotting dataset
print "Read Data!"
print "Plotting..."

fig = plt.figure()
#plt.subplot(2,1,1)
#plt.plot(power)
#plt.title('Full Year Power Price Data')
#plt.show()

for index in range(len(power) - TIME_LENGTH):
    data.append(power[index: index + TIME_LENGTH])

data = np.array(data)

datamean = data.mean()

## Shift by the mean to center around zero
data -= datamean
print "Shifted Data by: " + str(datamean)


## Separating data into Train and Test for LSTM
row = int(round(0.9 * data.shape[0]))
train = data[:row, :]
np.random.shuffle(train)

X_train = train[:, :-1]
y_train = train[:, -1]
X_test = data[row:, :-1]
y_test = data[row:, -1]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# How much data are we testing/ training with
print "lenth test" + str(len(X_test))
print "length train: " + str(len(train))

print "X Test: "
print X_test


## ----------- Defining Model -----------

model = Sequential()

model.add(LSTM(
    input_dim=LAYERS[0],
    output_dim=LAYERS[1],
    return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(
    LAYERS[2],
    return_sequences=False))

model.add(Dropout(0.2))

model.add(Dense(
        output_dim=LAYERS[3]))

## Linear activation since regression
model.add(Activation("linear"))

s = time.time()

## Calculating Loss with Mean Squared Error
model.compile(loss="mse", optimizer="rmsprop")

print "Compiled in: ", time.time() - s

## ----------- Running LSTM -----------

try:
    model.fit(
        X_train, y_train,
        batch_size=len(power)/TIME_LENGTH, nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT)

    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))

except KeyboardInterrupt:
    print 'Trained in: ', time.time() - s

'''
prediction_len = 24
prediction_seqs = []
predicted = [0]*prediction_len

initial_test = X_test[0]
print initial_test

## Predict hour then use that as estimate for next test
for i in range(0,24):
	predicted[i] = model.predict(initial_test)
'''

rmse = np.sqrt(((predicted[:TIME_LENGTH*7] - y_test[:TIME_LENGTH*7]) ** 2).mean())

print "MSE: " + str(rmse)
#print len(y_test)
#print len(predicted)

x_m = []
y_m = []

try:

    x = predicted[:TIME_LENGTH*7]
    y = y_test[:TIME_LENGTH*7]

    for i in x:
    	x_m.append(i + datamean)

    for i in y:
    	y_m.append(i + datamean)

    print x
    print y

    plt.title('Hourly Prediction using LSTM')
    plt.plot()
    plt.plot(x_m)
    plt.plot(y_m)

    #plt.plot(y)
    plt.legend(['Predicted Price','Actual Price'])

    plt.show()

except Exception as exception:
    print str(exception)