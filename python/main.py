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
#from statsmodels.tsa.statespace import SARIMAX
import statsmodels.api as sm
import statsmodels.tsa.stattools as st
import itertools

root = r'dataset/'

training_data = pandas.read_csv(root + 'training.csv')
training_price = training_data.loc[:,'price']
training_load= training_data.loc[:,'load']
#training_temp = training_data.loc[:,'temp']

''' Wavelet transform'''
db = pywt.Wavelet('db5')
[price,load] = [[],[]] # initialize wavelet transform vatiables

price = pywt.wavedec(training_price,db,level = 3)
load = pywt.wavedec(training_load,db,level =3)

''' step to estimate p,q,d for arima model?'''


#p = d = q = range(0, 4)
#pdq = list(itertools.product(p, d, q))
#seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]
#
##warnings.filterwarnings("ignore") # specify to ignore warning messages
#
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
#





