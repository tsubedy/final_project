import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
import time
import datetime as dt
from datetime import timedelta

from sklearn.metrics import mean_squared_error,r2_score
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
import pymongo

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Time Series forcasting App',
    layout='wide')

st.write("""
# Time Series Forcasting App 
**US COVID-19 cases** dataset is used to train the time series ARIMA model.
The following table shows the future forcasts of the number of days selected. Because of the datasize, it may take longer time to process the data if you choose too many days. 
""")

#---------------------------------#
# importing dataset
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["us_cases_db"]
collection = db.us_cases
cursor = collection.find()
cases_data = list(cursor)
us_cases_data = pd.DataFrame(cases_data, columns = ['submission_date', 'new_case'])

# Changing datatype for 'date'
us_cases_data["date"]=pd.to_datetime(us_cases_data["submission_date"])
us_cases_data = us_cases_data.drop(columns =['submission_date'])

# Changing datatype for new_case
us_cases_data = us_cases_data.astype({'new_case': 'float64'})
us_cases_data_datewise = us_cases_data.groupby(["date"]).agg({"new_case":'sum'})

us_cases_data_datewise["days_since_case"]= us_cases_data_datewise.index - us_cases_data_datewise.index[0]
us_cases_data_datewise["days_since_case"]= us_cases_data_datewise["days_since_case"].dt.days
us_cases_data_datewise['days_since_case'] = pd.DataFrame(us_cases_data_datewise, columns =['days_since_case'])

# Model building

X = us_cases_data_datewise.values[:,0]
train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:len(X)]

data = [x for x in train]
predictions = []

# walk-forward validation
for t in range(len(test)):
    model = ARIMA(data, order=(10,0,0)) # obtained the parameters from grid search
    model_fit = model.fit()
    output = model_fit.forecast()
    AR_pred = output[0]
    predictions.append(AR_pred)
    obs = test[t]
    data.append(obs)
    # st.write('predicted=%f, expected=%f' % (AR_pred, obs))

# evaluating predictions 
rmse = sqrt(mean_squared_error(test, predictions))
st.write('ARIMA Test RMSE: %.3f' % rmse)

new_date=[]
n = st.number_input("Enter the number of days (no decimal) to see the model forecast")

for i in range(1,int(n)+1):
    new_date.append(us_cases_data_datewise.index[-1]+timedelta(days=i))

pd.set_option('display.float_format', lambda x: '%.3f' % x)
model_forecast=pd.DataFrame(zip(new_date,predictions),
                               columns=["Dates","ARIMA forecast"])
st.table(model_forecast)