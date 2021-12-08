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

#---------------------------------#
# Model building
def build_model(arima):
    X = us_cases_data_datewise.values[:,0]
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:len(X)]
    data = [x for x in train]

    AR_predictions = []
    new_date = []
    # walk-forward validation
    for t in range(len(test)):
    model = ARIMA(data, order=(10,0,0)) # optimum value of p,d,q are determined using grid search
    model_fit = model.fit()
    output = model_fit.forecast()
    pred = output[0]
    AR_predictions.append(pred)
       
    for i in range(1, 15):
        new_date.append(us_cases_data_datewise.index[-1]+timedelta(days=i))
        AR_predictions.append(model_fit.forecast((len(test)+i)))
    print(AR_predictions)

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

st.write("""
# The Machine Learning App
The *ARIMA * model is used for building an algorithm. The forcasts for 15 days is displayed.
""")
build_model(arima)