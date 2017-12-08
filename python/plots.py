#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:43:22 2017

@author: Gautham
"""
import numpy as np
import matplotlib.pylab as plt


def acf_pacf_plot(data,cf):
    plt.subplot(121) 
    plt.plot(cf)
    plt.axhline(y=0,linestyle='--')
    plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--')
    plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--')