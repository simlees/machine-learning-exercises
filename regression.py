"""
REGRESSION - Take continuous data and figure out best fit

Supervised Learning - boils down to features (attributes/continuous data) and labels


"""

import pandas as pd
import numpy as np
import quandl
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = "7hZPqQqG7V3t1pLqvsRT"

df = quandl.get("WIKI/INTC")
df = df[['Adj. Open','Adj. Low','Adj. High','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100 # High/Low percentage
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Close'] * 100 # Percent Change
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True) # Fill NaN's
print(df.tail())

forecast_out = int(math.ceil(0.01*len(df))) # Integer - 10% length of data
# forecast_out = 10

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.tail())

X = np.array(df.drop(['label'],1)) # Features
y = np.array(df['label']) # Labels
X = preprocessing.scale(X)
y = np.array(df['label'])

# Helper method to split features (X) and labels (y) into training and test data (60/40) split. Test size is % of data used
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# clf = LinearRegression(n_jobs=-1) # n_jobs - no. of threads. -1 will run maximum
clf = svm.SVR() # Support Vector Regression
clf.fit(X_train, y_train) # Train
accuracy = clf.score(X_test, y_test)

print(accuracy)
