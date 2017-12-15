##Colin Vincent
##Support Vector Regression for Price data
##December 14, 2017

import os
import time
import numpy as np 
import pandas as pd 
import csv
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Reference: https://github.com/mKausthub/stock-er/blob/master/stock-er.py
# http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html

## ----------- Reading in Dataset -----------

file_name = 'dataset/training.csv'

with open(file_name) as file:
        dataset = csv.reader(file, delimiter=",")

        # Dataset has power and load info
        power = []
        load = []
        dates=[]

        num_values = 0

        for line in dataset:
            try:
                power.append(float(line[0]))
                dates.append(num_values)
                load.append(float(line[1]))
                num_values += 1

            except ValueError:
                print "Cound't pull data at index: " + str()

data = []

## Plotting dataset
print "Read Data!"
print "Plotting..."

#fig = plt.figure()
#plt.subplot(2,1,1)
#plt.plot(power)
#plt.title('Full Year Power Price Data')
#plt.show()

## ----------- SVR -----------

print power
print dates

print "Starting SVR!"
dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1
power_est = power[345*24:]
dates_est = dates[345*24:]
power = power[345*24:len(dates)-24]
dates = dates[345*24:len(dates)-24]

# SVR with RBF kernels
svr_rbf = SVR(kernel= 'rbf', C= 1000, gamma= 0.1)
svr_rbf.fit(dates, power)

plt.scatter(dates, power, color= 'black', label= 'Data')
plt.plot(dates_est, svr_rbf.predict(dates_est), color= 'red', label= 'RBF model')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()