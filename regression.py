"""
REGRESSION - Take continuous data and figure out best fit

Supervised Learning - boils down to features (attributes/continuous data) and labels


"""

import pandas as pd
import numpy as np
import quandl, datetime, math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = "7hZPqQqG7V3t1pLqvsRT"

df = quandl.get("WIKI/INTC")
df = df[['Adj. Open','Adj. Low','Adj. High','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100 # High/Low percentage
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Close'] * 100 # Percent Change
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True) # Fill NaN's

forecast_out = int(math.ceil(0.01*len(df))) # Integer - 10% length of data
# forecast_out = 10

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1)) # Features
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label']) # Labels
y = np.array(df['label'])

# Helper method to split features (X) and labels (y) into training and test data (60/40) split. Test size is % of data used
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# clf = LinearRegression(n_jobs=-1) # n_jobs - no. of threads. -1 will run maximum
clf = svm.SVR() # Support Vector Regression
clf.fit(X_train, y_train) # Train
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
