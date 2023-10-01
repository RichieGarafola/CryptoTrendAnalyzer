# Crypto Analysis Dashboard

## Overview
The Crypto Analysis Dashboard is a web-based application that allows users to analyze historical price data and perform various analyses on different cryptocurrencies (altcoins). This dashboard is designed to help users make informed decisions by providing insights into price trends, statistical metrics, price predictions, and more. Whether you're a seasoned trader or a novice looking to explore the cryptocurrency market, this dashboard provides a comprehensive suite of features to aid your decision-making process.

## Features

- Price Trends: Visualize historical price trends for selected altcoins.
        Gain a clear understanding of how the price of your chosen altcoin has evolved over time.


- Daily Returns: Calculate and visualize daily returns for the selected altcoin.
        Analyze the daily percentage changes in the price of your selected altcoin to assess its volatility and performance.


- Rolling Statistics: Compute rolling statistics like moving averages and standard deviations.
        Customize the window size to calculate moving averages and standard deviations, helping you identify trends and potential entry/exit points.


- Bollinger Bands: Generate Bollinger Bands charts for technical analysis.
        Utilize this classic technical analysis tool to assess price volatility and potential reversal points.


- Price Prediction: Utilize linear regression to predict future price trends.
        Predict future price trends based on historical data, allowing you to make informed investment decisions.


- Monte Carlo Simulation: Simulate price forecasts using a Monte Carlo method.
        Use the Monte Carlo method to create multiple price scenarios, aiding in risk assessment and long-term planning.


- Prophet Forecast: Use Facebook Prophet for time series forecasting.
        Leverage this robust time series forecasting library to make data-driven predictions about future price movements.


- Augmented Dickey-Fuller Test: Conduct a stationarity test on price data.
        Determine whether the price data of your selected altcoin exhibits stationarity, a critical consideration in time series analysis.


- Volume Analysis: Explore trading volume-related analyses including:
        - Trading Volume: Examine trading volume trends.
        - On-Balance Volume: Assess cumulative volume trends.
        - Volume Profile: Understand volume distribution across different price levels.
        - Relative Volume: Analyze trading volume relative to the average.
        - Volume Correlation Matrix: Explore the correlation between trading volumes of different altcoins.
        
## Usage
- Select an altcoin from the sidebar to begin analyzing its data.
- Choose an analysis option from the sidebar menu.
- Adjust any relevant parameters (e.g., rolling window size or simulation settings).
- Explore the visualizations and results displayed in the dashboard.

## Dependencies
- yfinance: Used to fetch historical cryptocurrency price data.
- pandas: Data manipulation and analysis.
- numpy: Numerical operations.
- matplotlib: Plotting charts and graphs.
- seaborn: Visualization enhancements.
- statsmodels: Time series analysis tools.
- streamlit: Web app framework for creating interactive dashboards.
- scikit-learn: Machine learning for price prediction.
- prophet: Time series forecasting library from Facebook.