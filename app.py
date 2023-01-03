import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime
from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()
import streamlit as st 


start="2013-01-01"
end="2022-12-31"


st.title("Stock Price Prediction")

user_input = st.text_input("Enter Stock Ticker", 'TSLA')
df = pdr.DataReader("user_input", start, end)

#describing data
st.subheader("Data from 2013-2022 i.e. of last decade")
st.write(df.describe())

#visualization
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

'''#splitting data into training and testing'''
data_training= pd.DataFrame(df.Close[0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df.Close[int((len(df)*0.70)):int(len(df))])

#scaling b/w 0 to 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#load model
model = load_model('tensor_model.h5')

past_100_days= data_training.tail(100)
final_testing = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_testing)

x_test = []
y_test= []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0]) ## 0 here bcoz of 1 columns(close) in df

x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)
scaler = scaler.scale_ #factor by all values are scale down

scle_factr = 1/scaler[0]
y_pred= y_pred*scle_factr
y_test = y_test*scle_factr


#final Visualization
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig2)