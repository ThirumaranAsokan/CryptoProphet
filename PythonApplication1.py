# imports 
import streamlit as st
from datetime import date

import yfinance as yf
import prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go 

START = "2015-01-01"
TODAY = date.today().strftime('%Y-%m-%d')

st.title("Crypto prediction app")

# A hardcoded preset of tickers, will add in searchbar for greater flexibility
stocks = ('BTC-USD','ETH-USD','USDT-USD','USDC-USD','BNB-USD','XRP-USD','BUSD-USD','ADA-USD','SOL-USD','DOT-USD','DOGE-USD','HEX-USD','AVAX-USD','DAI-USD',
'MATIC-USD','WTRX-USD','SHIB-USD','TRX-USD','ETC-USD','LEO-USD','LTC-USD')
selected_stock = st.sidebar.selectbox("Select dataset for prediction", stocks)

n_years = st.sidebar.slider("Years of prediction", 1, 4)
period = n_years*365

@st.cache
def load_data(ticker):
    """
    Load data via yahoo finance python API 
    """
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Display message on app to inform user of current state of data
data_load_state = st.sidebar.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting with  prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

m = prophet.Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast graph')
fig1 = plot_plotly(m, forecast,xlabel='Future Forecast',ylabel='Prediction')
st.plotly_chart(fig1)

st.write('forecast Prediction on all time basis')
fig2 = m.plot_components(forecast)
st.write(fig2)


st.write('AUTHOR: BY THIRUMARAN ASOKAN..')
