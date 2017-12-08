#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:56:08 2017

@author: Gautham


"""

import pandas
import pywt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pylab as plt
#from statsmodels.tsa.statespace import SARIMAX
import statsmodels.api as sm
import statsmodels.tsa.stattools as st
import itertools
import plots


root = r'dataset/'

training_data = pandas.read_csv(root + 'training.csv')
validation_data = pandas.read_csv(root + 'validation.csv')

training_price = training_data.iloc[:,0]
training_load= training_data.iloc[:,1]

validation_price = training_data.iloc[:,0]
validation_load = training_data.iloc[:,1]

#training_temp = training_data.loc[:,'temp']

''' Wavelet transform'''
db = pywt.Wavelet('db5')
[price,load] = [[],[]] # initialize wavelet transform vatiables

price = pywt.wavedec(training_price,db,level = 3)
load = pywt.wavedec(training_load,db,level =3)

''' step to estimate p,q,d for arima model?'''

#from statsmodels.tsa.stattools import acf, pacf
#lag_acf = acf(training_price.values, nlags = 2000)
#lag_pacf = pacf(training_price.values, nlags = 2000)
#plots.acf_pacf_plot(training_price,lag_acf)
#plots.acf_pacf_plot(price[0],lag_pacf)
from statsmodels.tsa.stattools import adfuller

#

''' ARIMA model '''

results = {}
mod = {}
predict = [0,0,0,0]

for i in range(0,4):
    mod[i] = sm.tsa.statespace.SARIMAX(price[i],order=[1,0,1],
                                            seasonal_order=[1,0,1,24],
                                            exog = load[i],
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
    results[i] = mod[i].fit()
    print results[i].summary()
    predict[i] = results[i].predict()
    
'''WT Reconstruction'''
    
arima_predix = pywt.waverec(predict,'db5')
plt.plot(arima_predix)
plt.plot(validation_price.values)
#

#p = d = q = range(1, 3)
#pdq = list(itertools.product(p, d, q))
#seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]

#warnings.filterwarnings("ignore") # specify to ignore warning messages

#for param in pdq:
#    for param_seasonal in seasonal_pdq:
#        try:
#            mod = sm.tsa.statespace.SARIMAX(price[0],
#                                            order=param,
#                                            seasonal_order=param_seasonal,
#                                            enforce_stationarity=False,
#                                            enforce_invertibility=False)
#
#            results = mod.fit()
#
#            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
##            print results.summary()
#        except:
#            continue
##





