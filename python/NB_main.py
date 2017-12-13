#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 22:51:08 2017

@author: Gautham
"""
import pandas
import NB_functions as nbf
import numpy as np
import matplotlib.pylab as plt


''' Initialize dataset '''

root = r'dataset/'

training_data = pandas.read_csv(root + 'training.csv')
validation_data = pandas.read_csv(root + 'validation.csv')

training_date_range = pandas.date_range(start= '01/01/2013 00:00:00',
                               end = '31/12/2014 23:00:00', freq = 'H')
validation_date_range = pandas.date_range(start= '01/01/2015 00:00:00',
                               end = '31/12/2015 23:00:00', freq = 'H')

training_data.index = training_date_range
validation_data.index = validation_date_range

''' Remove statistical price spikes >1.75*rolling mean'''


training_data = nbf.removeSpikes(training_data)

training_price = training_data.iloc[:,0]
training_load= training_data.iloc[:,1]

validation_price = validation_data.iloc[:,0]
validation_load = validation_data.iloc[:,1]


df_weekday = nbf.createDataFrame()
df_weekend = nbf.createDataFrame()
df_all = pandas.DataFrame()
[i,j] = [0,0]

''' TRAIN Store [price,demand] set from training data in a 24x12 matrix'''
timestep = training_data.index
df_wd = training_price[timestep.dayofweek <5]
df_we = training_price[timestep.dayofweek >4]

for month in range(1,13):
    for hour in range(0,24):
        df_weekday.values[hour,month-1] = df_wd[np.logical_and(df_wd.index.month==month,
                         df_wd.index.hour==hour)].values
        df_weekend.values[hour,month-1]= df_we[np.logical_and(df_we.index.month==month,
                         df_we.index.hour==hour)].values

''' PREDICTION BLOCK'''

predicted_price = pandas.DataFrame(index = validation_data.index,
                                   columns = ['price','probability'])
actual_price = pandas.DataFrame(index = validation_data.index,
                                   columns = ['price','probability'])
i=0
predicted_wd = predicted_price[predicted_price.index.dayofweek<5]
predicted_we = predicted_price[predicted_price.index.dayofweek>4]
error = [1]*8760

'''initialize weights'''

[wt1,wt2] = [0.5,0.5]

''' RUN prediction '''

for date in predicted_price.index:
    month = date.month
    hour = date.hour
    weekday = date.dayofweek
    df_demand = validation_load
    if weekday<5:
        predicted_price.values[i,0] = nbf.getSimpleMLE(df_weekday,
                       validation_load.values[i],month-1,hour)
    else:
        predicted_price.values[i,0] = nbf.getSimpleMLE(df_weekend,
                           validation_load.values[i],month-1,hour)
        
    ''' Get prediction based on observed demand '''
    
    p_demand = nbf.getPredictionBasedOnDemand(training_data,
                          df_demand.values[i])
    predicted_price.values[i,0] = (wt1*predicted_price.values[i,0] + wt2*p_demand)
    
    ''' Remove statistically aberrant predictions'''
    
    if predicted_price.values[i,0] >= 500:
        predicted_price.values[i,0] = 60
        
    '''Calculate error and adjust weights'''
    
    error[i] = (validation_price.values[i] - predicted_price.values[i,0])
    if error[i] > 0: 
        wt1 +=0.01
    elif error[i] < 0:
        wt1 -=0.01
        
    ''' Store observed values back into memory'''
    
    df_weekday,df_weekend = nbf.updateHistoricValues(validation_price.values[i],
                                                     validation_price.index[i],
                                                     df_weekday,df_weekend)
    i+=1
    
plt.plot(predicted_price.values[:,0])
plt.plot(validation_price.values[:])
NB_rmse = np.sqrt(((predicted_price.values[:,0] - validation_price.values) ** 2).mean())
        
        

    
    
    


