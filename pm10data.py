#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:24:07 2019
Univariate and multivariate time series analysis with Python
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing as pp
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf


data = pd.read_csv("pm10data.csv")
data.columns
data.shape
data.info()
data.head()

# descriptive statistics
data.describe()

# boxplot
data.boxplot(figsize=(15, 6),grid=False) 
# histogram
data.hist(figsize=(15, 6),bins=30, sharex=True, sharey=True,normed=True,grid=False) 

# time series splot
data.plot(subplots=True, figsize=(15,15))

# scaling to calculate normalized ACFs
ndata = data
ndata[data.columns] = pp.scale(data[data.columns])

# autocorrelation and cross correlation plots
for i in range(data.shape[1]):
    for j in range(data.shape[1]):
        if i >= j:
            plt.xcorr(ndata[data.columns.values[i]], ndata[data.columns.values[j]],maxlags=40)
            plt.suptitle(data.columns.values[i] + " x " + data.columns.values[j])
            plt.show()
            
# data present seasonality of period=7

# ADF test for stationarity       
for i in range(data.shape[1]):
    result = adfuller(ndata[data.columns.values[i]])
    print('ADF test for {} \n\t ADF test statistic: {} \n\t p-value: {} '.format(data.columns.values[i], result[0], result[1]))


# fitting a VAR(p)
model = VAR(data)

# getting VAR(1) residuals
fittedmodel = model.fit(maxlags=1)
residuals = fittedmodel.resid

# autocorrelation and cross-correlation plots of residuals
for i in range(data.shape[1]):
    for j in range(data.shape[1]):
        if i >= j:
            plt.figure(figsize=(10,10)) 
            plt.xcorr(residuals[data.columns.values[i]], residuals[data.columns.values[j]],maxlags=40)
            plt.suptitle(data.columns.values[i] + " x " + data.columns.values[j])
            plt.show()

# as expected, residuals present seasonality of period=7
# should consider a SVARMA or maybe a higher order model
            
# getting VAR(7) residuals
fittedmodel = model.fit(maxlags=7)
residuals = fittedmodel.resid

# autocorrelation and cross-correlation plots of residuals
for i in range(data.shape[1]):
    for j in range(data.shape[1]):
        if i >= j:
            plt.figure(figsize=(10,10)) 
            plt.xcorr(residuals[data.columns.values[i]], residuals[data.columns.values[j]],maxlags=40)
            plt.suptitle(data.columns.values[i] + " x " + data.columns.values[j])
            plt.show()
            
# VAR(7) seems to filter well the series even with the seasonaliry
            

# checking a VAR(1) with deseasonalized data
# adjusting individual seasonal adjustment
des_data = data
for i in data.columns:
    des_data[i] = (data[i] - seasonal_decompose(data[i],freq=7).seasonal)
        
            
# fitting a VAR(p) with deseasonalized data
model = VAR(des_data)

# getting VAR(1) residuals
fittedmodel = model.fit(maxlags=1)
residuals = fittedmodel.resid  

for i in range(data.shape[1]):
    for j in range(data.shape[1]):
        if i >= j:
            plt.figure(figsize=(10,10)) 
            plt.xcorr(residuals[data.columns.values[i]], residuals[data.columns.values[j]],maxlags=40)
            plt.suptitle(data.columns.values[i] + " x " + data.columns.values[j])
            plt.show()  
            
# similar results as in R