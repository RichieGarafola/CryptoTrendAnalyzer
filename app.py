# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from prophet import Prophet
from datetime import datetime, timedelta

# Streamlit interface
st.title("Crypto Analysis Dashboard")

# Define a list of altcoin symbols you want to track
# Example altcoins: Bitcoin, Ethereum, Cardano, Ripple, Litecoin
altcoins = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "LTC-USD"]  

# Create a dictionary to store historical data for each altcoin
altcoin_data = {}

# Function to fetch historical data for a given altcoin and date range
def fetch_altcoin_data(altcoin_symbol, start_date, end_date):
    altcoin_ticker = yf.Ticker(altcoin_symbol)
    # Fetch data for the specified date range
    altcoin_data[altcoin_symbol] = altcoin_ticker.history(start=start_date, end=end_date)

# Sidebar selection for altcoin
selected_altcoin = st.sidebar.selectbox("Select Altcoin", altcoins)

# Allow the user to select the start date
start_date = st.sidebar.date_input("Select Start Date", datetime(2020, 1, 1))

# Allow the user to select the end date (default to today's date)
end_date = st.sidebar.date_input("Select End Date", datetime.today())

# Fetch historical data for the selected altcoin within the chosen date range
fetch_altcoin_data(selected_altcoin, start_date, end_date)
selected_data = altcoin_data[selected_altcoin]

# Display Altcoin Price Trends
st.subheader(f"{selected_altcoin} Price Trends")
selected_data = altcoin_data[selected_altcoin]
st.line_chart(selected_data['Close'])

# Sidebar options for analysis
analysis_option = st.sidebar.radio("Choose Analysis", ["Daily Returns", "Rolling Statistics", "Bollinger Bands", "Price Prediction", "Monte Carlo Simulation", "Prophet Forecast", "Augmented Dickey-Fuller Test", "Volume Analysis"])

if analysis_option == "Daily Returns":
    # Calculate daily returns for the selected altcoin
    selected_data['Daily Returns'] = selected_data['Close'].pct_change()
    st.subheader(f"{selected_altcoin} Daily Returns")
    st.line_chart(selected_data['Daily Returns'])
    
elif analysis_option == "Rolling Statistics":
    # Calculate rolling statistics (e.g., moving average and standard deviation) for altcoin prices
    window = st.sidebar.slider("Select Rolling Window", 1, 500, 20)
    selected_data[f'{window}-Day MA'] = selected_data['Close'].rolling(window=window).mean()
    selected_data[f'{window}-Day Std'] = selected_data['Close'].rolling(window=window).std()
    st.subheader(f"{selected_altcoin} Rolling Statistics")
    st.line_chart(selected_data[[f'{window}-Day MA', f'{window}-Day Std']])
    
elif analysis_option == "Bollinger Bands":
    # Calculate Bollinger Bands for the selected altcoin
    window = st.sidebar.slider("Select Bollinger Bands Window", 1, 100, 30)
    selected_data[f'{window}-Day MA'] = selected_data['Close'].rolling(window=window).mean()
    selected_data[f'{window}-Day Std'] = selected_data['Close'].rolling(window=window).std()
    selected_data['Upper Band'] = selected_data[f'{window}-Day MA'] + (2 * selected_data[f'{window}-Day Std'])
    selected_data['Lower Band'] = selected_data[f'{window}-Day MA'] - (2 * selected_data[f'{window}-Day Std'])
    st.subheader(f"{selected_altcoin} Bollinger Bands")
    st.line_chart(selected_data[['Close', 'Upper Band', 'Lower Band']])
    
elif analysis_option == "Price Prediction":
    # Machine Learning: Linear Regression for Price Prediction
    st.subheader(f"{selected_altcoin} Price Prediction")

    # Data preparation
    selected_data = altcoin_data[selected_altcoin].dropna()
    X = np.arange(len(selected_data)).reshape(-1, 1)
    y = selected_data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict prices
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Plot the actual vs. predicted prices
    # Create a Matplotlib figure and axis within Streamlit
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"{selected_altcoin} Price Prediction")
    ax.plot(selected_data.index[-len(X_test):], y_test, label="Actual Price", color='blue', linestyle='--')
    ax.plot(selected_data.index[-len(X_test):], y_pred, label="Predicted Price", color='red')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid()

    # Display the Matplotlib figure within Streamlit
    st.pyplot(fig)

    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    
elif analysis_option == "Monte Carlo Simulation":
    # Monte Carlo Simulation for Price Forecasting
    st.subheader(f"{selected_altcoin} Price Forecast (Monte Carlo Simulation)")

    # Data preparation for simulation
    last_price = selected_data['Close'].iloc[-1]

    # Check if 'Daily Returns' column exists, if not, calculate it
    if 'Daily Returns' not in selected_data.columns:
        selected_data['Daily Returns'] = selected_data['Close'].pct_change()

    # Continue with the rest of the code
    volatility = selected_data['Daily Returns'].std()
    
    # Number of simulations and days
    num_simulations = st.sidebar.slider("Number of Simulations", 1, 1000, 100)
    num_days = st.sidebar.slider("Number of Days to Forecast", 1, 365, 30)

    # Monte Carlo simulation
    simulation_df = pd.DataFrame()
    for i in range(num_simulations):
        daily_returns = np.random.normal(0, volatility, num_days) + 1
        price_series = [last_price]
        for j in range(num_days):
            price_series.append(price_series[-1] * daily_returns[j])
        simulation_df[f'Simulation {i+1}'] = price_series

    # Visualize Monte Carlo simulations using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"{selected_altcoin} Price Forecast (Monte Carlo Simulation)")
    for i in range(num_simulations):
        ax.plot(simulation_df.index, simulation_df[f'Simulation {i+1}'], lw=1)
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.grid()

    # Display the Matplotlib figure within Streamlit
    st.pyplot(fig)
    
elif analysis_option == "Prophet Forecast":
    # Prophet Forecasting
    st.subheader(f"{selected_altcoin} Price Forecast (Prophet Forecast)")


    # Prepare the data for Prophet
    df_prophet = selected_data[['Close']].reset_index()
    df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

    # Create a Prophet model
    model_prophet = Prophet(yearly_seasonality=True)

    # Remove timezone from the 'ds' column
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    # Fit the model with altcoin data
    model_prophet.fit(df_prophet)

    # Forecast future data
    future_prophet = model_prophet.make_future_dataframe(periods=52, freq="W")
    forecast_prophet = model_prophet.predict(future_prophet)

    # Plot predictions
    st.subheader(f"{selected_altcoin} Prophet Forecast Predictions")
    fig, ax = plt.subplots()
    ax.plot(forecast_prophet['yhat'], label='yhat', color='blue')
    ax.fill_between(
        forecast_prophet.index,
        forecast_prophet['yhat_lower'],
        forecast_prophet['yhat_upper'],
        color='gray',
        alpha=0.2,
        label='yhat_lower and yhat_upper'
    )
    ax.legend()

    # Display the Prophet forecast plot
    st.pyplot(fig)

    # Display Prophet forecast components
    st.subheader(f"{selected_altcoin} Prophet Forecast Components")
    fig_components = model_prophet.plot_components(forecast_prophet)
    st.pyplot(fig_components)

    # Display 21-day forecast
    st.subheader("21 Day Forecast")
    forecast_21_days = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(21)
    forecast_21_days.rename(columns={'ds': 'Date', 'yhat': 'Most Likely Case', 'yhat_lower': 'Worst Case', 'yhat_upper': 'Best Case'}, inplace=True)
    st.write(forecast_21_days)
    

# Augmented Dickey-Fuller Test
elif analysis_option == "Augmented Dickey-Fuller Test":
    result = adfuller(selected_data['Close'], autolag='AIC')
    st.subheader("Augmented Dickey-Fuller Test for Stationarity")
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"P-value: {result[1]:.4f}")
    st.write("Critical Values:")
    for key, value in result[4].items():
        st.write(f"  {key}: {value:.4f}")
    if result[1] <= 0.05:
        st.write("Stationary (Reject null hypothesis)")
    else:
        st.write("Non-Stationary (Fail to reject null hypothesis)")
    
elif analysis_option == "Volume Analysis":
    # Sidebar options for volume-related analysis
    volume_analysis_option = st.sidebar.radio("Select Volume Analysis", ["Trading Volume", "On-Balance Volume", "Volume Profile", "Relative Volume", "Volume Correlation Matrix"])

    # Add volume-related analysis options
    if volume_analysis_option == "Trading Volume":
        st.subheader(f"{selected_altcoin} Trading Volume")
        st.line_chart(selected_data['Volume'])
    elif volume_analysis_option == "On-Balance Volume":
        # Calculate OBV
        selected_data['OBV'] = np.where(selected_data['Close'] > selected_data['Close'].shift(1), selected_data['Volume'], 
                                        np.where(selected_data['Close'] < selected_data['Close'].shift(1), -selected_data['Volume'], 0))
        selected_data['OBV'] = selected_data['OBV'].cumsum()
        st.subheader(f"{selected_altcoin} On-Balance Volume (OBV)")
        st.line_chart(selected_data['OBV'])
    elif volume_analysis_option == "Volume Profile":
        # Allow the user to select the price interval
        price_interval = st.slider("Select Price Interval for Volume Profile", 1, 100, 10)

        # Calculate volume profile
        price_bins = np.arange(selected_data['Close'].min(), selected_data['Close'].max(), price_interval)
        volume_profile = []
        for price_level in price_bins:
            volume_at_level = selected_data[(selected_data['Close'] >= price_level) & (selected_data['Close'] < price_level + price_interval)]['Volume'].sum()
            volume_profile.append(volume_at_level)

        # Display the volume profile chart
        st.subheader(f"{selected_altcoin} Volume Profile")
        fig, ax = plt.subplots()
        plt.bar(price_bins, volume_profile, width=price_interval)
        plt.xlabel("Price Levels")
        plt.ylabel("Volume")
        st.pyplot(fig)
    elif volume_analysis_option == "Relative Volume":
        # Calculate the average volume
        average_volume = selected_data['Volume'].mean()

        # Calculate relative volume
        selected_data['Relative Volume'] = selected_data['Volume'] / average_volume

        # Display relative volume chart
        st.subheader(f"{selected_altcoin} Relative Volume")
        st.line_chart(selected_data['Relative Volume'])
    elif volume_analysis_option == "Volume Correlation Matrix":
        # Create a DataFrame to store altcoin volumes
        volume_data = pd.DataFrame()

        # Fetch volumes for all altcoins and store in the DataFrame
        for altcoin_symbol, altcoin_history in altcoin_data.items():
            volume_data[altcoin_symbol] = altcoin_history['Volume']

        # Calculate the correlation matrix
        correlation_matrix = volume_data.corr()

        # Display the correlation matrix as a heatmap using Seaborn
        st.subheader("Altcoin Volume Correlation Matrix")
        plt.figure(figsize=(8, 6))
        plt.title("Volume Correlation Matrix")
        sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, square=True, cbar_kws={"shrink": 0.75})
        st.pyplot(plt.gcf())