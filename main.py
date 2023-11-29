# pip install streamlit fbprophet yfinance plotly

import streamlit as st
from datetime import date

import yfinance as yf
from plotly import graph_objs as go 
from prophet import Prophet
from prophet.plot import plot_plotly

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")
stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select dataset for prediction ", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365


# load stock data using yahoo-finance
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())


# plot data using plotly (go)
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


# forecasting using facebook-prophet model
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"}) # parameter name in docmtn.

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period) # req. in no.of days
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())


# plot forecast using fbprophet.plot (plot_plotly)
st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)


