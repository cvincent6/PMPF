#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:16:23 2017

@author: Gautham
"""
import pandas
import numpy as np
import openturns as ot

def createDataFrame():
    _index = range(0,24)
    _cols = range(1,13)
    '''Weekday - Month vs hour dataframe,
    Weekend - Month vs hour dataframe'''
    return pandas.DataFrame(index = _index, columns = _cols )

def getNearestDemand(array,value):
    index = np.abs(array[:]-value).argmin()
    return array[index]

def getMaximumLikelihood(df,demand,month,hour):
    '''Get Kernel model for given month,hour block'''
    _model = ot.Sample(df.values[hour,month])
    _kernel = ot.KernelSmoothing()
    k = _kernel.build(_model)
    [prob,prev_prob,maxima] = [0,0,0]
    for price in np.arange(10,500,1):
        prob = k.computePDF([price,5100])
        if prob>prev_prob:
            maxima = [price,prob]
        prev_prob = prob
    return maxima

def getSimpleMLE(df,demand,month,hour):
    p_set = df.values[hour,month]
    return np.mean(p_set)

def getPredictionBasedOnDemand(df,demand):
    p_set = df[np.logical_and(df.values[:,1]>demand-100,
                              df.values[:,1]<demand+100)].values[:,0]

    p_sing = df[df.values[:,1]==demand].values[:,0]
    sing_median = np.median(p_sing)
    set_median = np.mean(p_set)
    if np.isnan(set_median):
        print 'here'
        set_median = getNearestDemand(df.values[:,1],demand)
    return set_median
    
    
def removeSpikes(df):
    rolling_mean = df.resample('M').mean()
    i = 0
    for date in df.index:
        if(df.values[i,0] > 1.75*rolling_mean.values[date.month-1,0]):
            df.values[i,0] = rolling_mean.values[date.month-1,0]
        i+=1
    return df

def updateHistoricValues(data,index,df_weekday,df_weekend):
    if index.dayofweek <5:
        np.append(df_weekday.values[index.hour,index.month-1],data)
    else:
        np.append(df_weekend.values[index.hour,index.month-1],data)
    return df_weekday,df_weekend
        
        