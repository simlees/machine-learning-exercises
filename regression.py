"""
REGRESSION - Take continuous data and figure out best fit

Supervised Learning - boils down to features (attributes/continuous data) and labels


"""

import pandas as pd
import quandl
import math

quandl.ApiConfig.api_key = "7hZPqQqG7V3t1pLqvsRT"
df = quandl.get("WIKI/INTC")

df = df[['Adj. Open','Adj. Low','Adj. High','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100 # High/Low percentage

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Close'] * 100 # Percent Change

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True) # Fill NaN's
print(df.tail())

forecast_out = int(math.ceil(0.01*len(df))) # Forecast 10% length of data

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.tail())
