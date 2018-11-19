# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 13:54:46 2018

@author: Sanmoy
"""

import numpy as np
import pandas as pd

import os
os.getcwd()
os.chdir("C:\\F\\NMIMS\\DataScience\\Sem-3\\DL\\data")

pnbData = pd.read_csv("PNB.NS.csv")
pnbData.head(5)

prediction_days = 100

df_train= pnbData[:len(pnbData)-prediction_days]
df_test= pnbData[len(pnbData)-prediction_days:]

df_train.shape
df_test.shape

training_set = df_train.iloc[:, 5:6].values
test_set = df_test.iloc[:, 5:6].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
X_train=training_set_scaled[0:len(df_train)-1]
y_train=training_set_scaled[1:len(df_train)]


inputs = sc.transform(test_set)
inputs = np.reshape(inputs, (df_test.shape[0], 1, 1))


X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, activation='relu', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 200, batch_size = 8)


###### Prediction for next 30 days ############
testPred_closePrice=regressor.predict(inputs)
testPred_closePrice=sc.inverse_transform(testPred_closePrice)

import matplotlib.pyplot as plt
predDF = pd.DataFrame(np.concatenate((np.array(df_test.iloc[:, 0:1].values), df_test.iloc[:, 5:6], testPred_closePrice), axis=1), columns=['Date', 'ActualClosePrice', 'PredictedClosePrice'])
predDF.set_index('Date', inplace=True)
predDF.plot()


####### Train Prediction ############
trainPred_closePrice=regressor.predict(X_train)
trainPred_closePrice=sc.inverse_transform(trainPred_closePrice)

predDF = pd.DataFrame(np.concatenate((np.array(df_train.iloc[1:, 0:1].values), df_train.iloc[1:, 5:6], trainPred_closePrice), axis=1), columns=['Date', 'ActualClosePrice', 'PredictedClosePrice'])
predDF.set_index('Date', inplace=True)
predDF.plot()

