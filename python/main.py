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

root = r'dataset/'

training_data = pandas.read_csv(root + 'training.csv')
training_price = training_data.loc[:,'price']
training_load= training_data.loc[:,'load']
#training_temp = training_data.loc[:,'temp']

''' Wavelet transform'''
db = pywt.Wavelet('db5')
[p,l] = [[],[]] # initialize wavelet transform vatiables

p = pywt.wavedec(training_price,db,level = 3)
l = pywt.wavedec(training_load,db,level =3)

''' step to estimate p,q,d for arima model?'''







