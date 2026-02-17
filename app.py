import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page Config
st.set_page_config(page_title="Mouna Loa CO2 Prediction", layout="wide")

# Title and Description
st.title("Mouna Loa CO2 Prediction Analysis")
st.markdown("""
This application analyzes CO2 levels from the Mouna Loa dataset and forecasts future values using a SARIMAX model.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload archive.csv", type=['csv'])

if uploaded_file is not None:
    # Load Data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Preprocessing (following the notebook logic)
        df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
        df.set_index('Date', inplace=True)
        df.index.freq = 'MS'
        
        # Handling Missing Values (as done in the notebook)
        df['Carbon Dioxide (ppm)'] = df['Carbon Dioxide (ppm)'].replace(-99.99, np.nan) 
        df = df.fillna(method='bfill')
        df = df.fillna(method='ffill')
        df['co2'] = df['Carbon Dioxide (ppm)']

        # Sidebar Options
        st.sidebar.header("Analysis Settings")
        forecast_steps = st.sidebar.number_input("Forecast Steps (Months)", min_value=12, max_value=120, value=24)

        # 1. Visualization
        st.subheader("1. Time Series Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['co2'], label='CO2 Levels')
        ax.set_title("Mouna Loa CO2 Levels Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("CO2 (ppm)")
        ax.legend()
        st.pyplot(fig)

        # 2. Seasonal Decomposition
        st.subheader("2. Seasonal Decomposition")
        decomposition = seasonal_decompose(df['co2'], model='additive', period=12)
        fig_decomp = decomposition.plot()
        fig_decomp.set_size_inches(14, 10)
        st.pyplot(fig_decomp)

        # 3. Stationarity Test
        st.subheader("3. Stationarity Test (Augmented Dickey-Fuller)")
        adh_result = adfuller(df['co2'])
        st.write(f"ADF Statistic: {adh_result[0]}")
        st.write(f"p-value: {adh_result[1]}")
        if adh_result[1] < 0.05:
            st.success("The time series is stationary.")
        else:
            st.warning("The time series is non-stationary (common for CO2 data).")

        # 4. Model Training (SARIMAX)
        st.subheader("4. SARIMAX Model Forecasting")
        with st.spinner('Training SARIMAX(1,1,1)x(1,0,1,12) model...'):
            # Using the parameters identified in the notebook
            train_size = int(len(df) * 0.8)
            train, test = df.iloc[:train_size], df.iloc[train_size:]
            
            model = SARIMAX(train['co2'], 
                            order=(1, 1, 1), 
                            seasonal_order=(1, 0, 1, 12))
            results = model.fit(disp=False)
            
            # Forecasting
            forecast = results.get_forecast(steps=len(test))
            predicted_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()

            # Future Forecast
            future_model = SARIMAX(df['co2'], 
                                   order=(1, 1, 1), 
                                   seasonal_order=(1, 0, 1, 12))
            future_results = future_model.fit(disp=False)
            future_forecast = future_results.get_forecast(steps=forecast_steps)
            future_mean = future_forecast.predicted_mean
            future_conf = future_forecast.conf_int()

        # Plotting Results
        fig_forecast, ax = plt.subplots(figsize=(14, 7))
        ax.plot(train.index, train['co2'], label='Training Data')
        ax.plot(test.index, test['co2'], label='Actual Test Data', color='green')
        ax.plot(test.index, predicted_mean, label='Test Prediction', color='red', linestyle='--')
        
        # Plot Future
        future_index = pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='MS')[1:]
        ax.plot(future_index, future_mean, label='Future Forecast', color='orange')
        ax.fill_between(future_index, future_conf.iloc[:, 0], future_conf.iloc[:, 1], color='orange', alpha=0.1)

        ax.set_title("SARIMAX Forecast vs Actuals")
        ax.legend()
        st.pyplot(fig_forecast)

        # 5. Model Evaluation
        st.subheader("5. Model Evaluation Metrics (Test Set)")
        mae = mean_absolute_error(test['co2'], predicted_mean)
        rmse = np.sqrt(mean_squared_error(test['co2'], predicted_mean))
        
        col1, col2 = st.columns(2)
        col1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
        col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
        st.write("Please ensure the CSV has columns: 'Year', 'Month', 'Carbon Dioxide (ppm)'")

else:
    st.info("Awaiting CSV file upload. Please upload `archive.csv` to proceed.")
