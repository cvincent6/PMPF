## Colin Vincent
## Machine Learning 
## LSTM for Power Market Price Prediction
## December 14, 2017

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
EPOCHS = 5
RATIO = 0.5
LAYERS = [1, 50, 100, 1]
VALIDATION_SPLIT = 0.05

## ----------- Reading in Dataset -----------

file_name = 'dataset/training.csv'
fig = plt.figure()

with open(file_name) as file:
		dataset = csv.reader(file, delimiter=",")

		# Dataset has power and load info
		power_original = []
		load = []

		num_values = 0

		for line in dataset:
			try:
				power_original.append(float(line[0]))
				load.append(float(line[1]))
				num_values += 1
			except ValueError:
				print "Cound't pull data at index: " + str()

x_total = []
y_total = []
mse = []
power = []

for week in range(0,1):
	data = []
	power = power_original[:len(power_original)-24*(7-week)]
	## Plotting dataset
	print "Read Data!"
	print "length: " + str(len(power))

	#plt.subplot(2,1,1)
	#plt.plot(power)
	#plt.title('Full Year Power Price Data')
	#plt.show()

	for index in range(len(power) - TIME_LENGTH):
		data.append(power[index: index + TIME_LENGTH])

	data = np.array(data)

	## Shift by the mean to center around zero
	data -= data.mean()
	print "Shifted Data by: " + str(data.mean())


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

	rmse = np.sqrt(((predicted[:TIME_LENGTH] - y_test[:TIME_LENGTH]) ** 2).mean())
	mse.append(rmse)

	print "MSE: " + str(rmse)
	#print len(y_test)
	#print len(predicted)

	try:
		
		print predicted
		x = predicted[:TIME_LENGTH]
		y = y_test[:TIME_LENGTH]
		mean = data.mean()

		#x_total.append(x)
		#y_total.append(y)

		predicted2 = [x+1 for x in x]
		y_test2 = [y+1 for y in y]

		print y
		print x

		#plt.title('24 Hour Prediction using LSTM')
		#plt.plot()
		#plt.plot(predicted2)
		#plt.plot(y_test2)
		#plt.legend(['Predicted Price','Actual Price'])

		#plt.show()

	except Exception as exception:
		print str(exception)

plt.title('24 Hour Prediction using LSTM')
plt.plot(x)
plt.plot(y)
plt.legend(['Predicted Price','Actual Price'])
plt.show()
