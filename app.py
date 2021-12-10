import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')
import time
import datetime as dt
from datetime import timedelta

from sklearn.metrics import mean_squared_error,r2_score
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
import pymongo
import base64
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Time Series forecasting App',
    layout='wide')

st.write(""" # Time Series Forecasting App""") 
st.write(""" ## **US COVID-19 Cases and Death** """)

st.sidebar.title("Select Dataset ")
st.markdown('<style>body{background-color: lightblue;}<style>', unsafe_allow_html=True)

def load_data():
    # importing dataset
    client = pymongo.MongoClient("mongodb+srv://tsubedy:TS24751@cluster1.ppbek.mongodb.net/?retryWrites=true&w=majority")
    db = client["us_cases_db"]
    collection = db.us_cases
    cursor = collection.find()
    cases_data = list(cursor)
    us_cases_data = pd.DataFrame(cases_data, columns = ['submission_date', 'new_case', 'new_death'])

    # Changing datatype for 'date'
    us_cases_data["date"]=pd.to_datetime(us_cases_data["submission_date"])
    us_cases_data = us_cases_data.drop(columns =['submission_date'])

    # Changing datatype for new_case
    us_cases_data = us_cases_data.astype({'new_case': 'float64', 'new_death': 'float64'})
    us_data_datewise = us_cases_data.groupby(["date"]).agg({"new_case":'sum', "new_death":'sum'})

    us_data_datewise["days_since_case"]= us_data_datewise.index - us_data_datewise.index[0]
    us_data_datewise["days_since_case"]= us_data_datewise["days_since_case"].dt.days
    us_data_datewise['days_since_case'] = pd.DataFrame(us_data_datewise, columns =['days_since_case'])

    return us_data_datewise

us_data_datewise = load_data()

# Selection options
datasets = st.sidebar.selectbox('Forecasting for', ('Cases_data', 'Death_data' ))

if datasets == 'Cases_data':
    X = us_data_datewise['new_case']
elif datasets == 'Death_data':
    X = us_data_datewise['new_death']

# Model building
# X = case_data.values[:,0]

train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:len(X)]
data = [x for x in train]
predictions = []


# walk-forward validation
for t in range(len(test)):
    model = ARIMA(data, order=(10,2,2)) # obtained the parameters from grid search
    model_fit = model.fit()
    output = model_fit.forecast()
    AR_pred = output[0]
    predictions.append(AR_pred)
    obs = test[t]
    data.append(obs)
    # st.write('predicted=%f, expected=%f' % (AR_pred, obs))

# Plotting the predicted against test data

fig=px.scatter(x=test, y=predictions)
fig.update_layout(title="ARIMA Predicton vs Test", width=500, height=300)

st.plotly_chart(fig)


# evaluating predictions 
rmse = sqrt(mean_squared_error(test, predictions))
st.write('ARIMA Test RMSE: %.3f' % rmse)

new_date=[]
# n = st.number_input("Enter number of days (no decimal) to get the model forecast")

n = st.sidebar.number_input('Enter number of days to forecast', min_value=1, max_value = None, )

for i in range(1,int(n)+1):
    new_date.append(us_data_datewise.index[-1]+timedelta(days=i))

pd.set_option('display.float_format', lambda x: '%.3f' % x)
model_forecast=pd.DataFrame(zip(new_date,predictions),
                               columns=["Dates","ARIMA forecast"])
st.table(model_forecast)

st.sidebar.write("""
####  Download the Forecast Data
The link below allows you to download the newly created forecast data, as csv file 
""")
csv_exp = model_forecast.to_csv(index=False)
b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> '
st.sidebar.markdown(href, unsafe_allow_html=True)