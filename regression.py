"""
REGRESSION - Take continuous data and figure out best fit

Supervised Learning - boils down to features (attributes/continuous data) and labels


"""

import pandas as pd
import quandl

quandl.ApiConfig.api_key = "7hZPqQqG7V3t1pLqvsRT"
df = quandl.get("WIKI/INTC")

print(df.head())
