#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 22:51:08 2017

@author: Gautham
"""
import pandas


root = r'dataset/'

training_data = pandas.read_csv(root + 'training.csv')
validation_data = pandas.read_csv(root + 'validation.csv')

date_range = pandas.date_range(start= '01/01/1990 00:00:00',
                               end = '31/12/1990 23:00:00', freq = 'H')
training_data.index = date_range


training_price = training_data.iloc[:,0]
training_load= training_data.iloc[:,1]

validation_price = training_data.iloc[:,0]
validation_load = training_data.iloc[:,1]

weekday = range(0,6)
weekend = [6,7]
week_day_raw = []
week_end_raw = []

for timestep in training_price.index:
    for month in range(0,12):
        # week_day_raw[month] = training_price[timestep.month==month and timestep.day in weekday]
        week_day_raw= training_price[timestep.month==month and timestep.day in weekday]
#        week_end_raw[month] = training_price[timestep.month==month and timestep.day in weekend]
        
        



