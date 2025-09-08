import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

# Load the model and scalers
multivariate_lstm = joblib.load('multivariate_lstm_model.joblib')
scaler = joblib.load('feature_scaler.joblib')
target_scaler = joblib.load('target_scaler.joblib')

# Define the prediction function
def forecast_future_days(model, feature_scaler, target_scaler, last_window_data, num_days_to_forecast, window_size):
    forecasted_data = []
    current_window = last_window_data.copy()

    for _ in range(num_days_to_forecast):
        # Prepare the input data for the model
        input_data = feature_scaler.transform(current_window.values)
        input_data = np.array(input_data).reshape(1, window_size, input_data.shape[1])

        # Predict the next day's Open and Close prices
        predicted_scaled_values = model.predict(input_data)

        # Inverse transform the predicted Close price
        predicted_close = target_scaler.inverse_transform(predicted_scaled_values[:, 1].reshape(-1, 1))[0][0]

        # Inverse transform the predicted Open price (using the feature scaler, assuming Open was scaled with features)
        # Note: This assumes 'Open' was scaled as part of the features and is the first column in the scaled data.
        # You might need to adjust the index [0] if 'Open' is at a different position or scaled differently.
        predicted_open_scaled = predicted_scaled_values[:, 0].reshape(-1, 1)
        # Create a dummy array with the same number of features as the training data
        dummy_array = np.zeros((predicted_open_scaled.shape[0], feature_scaler.n_features_in_))
        # Place the scaled Open predictions into the correct column (assuming 'Open' is the first feature)
        dummy_array[:, 0] = predicted_open_scaled[:, 0]
        predicted_open = feature_scaler.inverse_transform(dummy_array)[:, 0][0]


        # For simplicity, use the predicted Close price as a placeholder for High and Low
        predicted_high = predicted_close
        predicted_low = predicted_close
        predicted_adj_close = predicted_close # Assuming Adj Close is close to Close for forecasting
        predicted_volume = current_window['Volume'].mean() # Use the average volume from the window as a placeholder

        # Create a DataFrame for the predicted day
        predicted_day_df = pd.DataFrame([[predicted_open, predicted_high, predicted_low, predicted_close, predicted_adj_close, predicted_volume]],
                                        columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

        # Append the predicted day's data to the forecast list
        forecasted_data.append(predicted_day_df.iloc[0])

        # Update the last available data window by removing the oldest and adding the new prediction
        # For the next iteration, we use the predicted values as the new last row
        current_window = pd.concat([current_window.iloc[1:], predicted_day_df], ignore_index=True)


    return pd.DataFrame(forecasted_data)

# Streamlit App
st.title('Apple Stock Price Forecast')

st.sidebar.header('Forecast Settings')
num_days_to_forecast = st.sidebar.number_input('Number of days to forecast (up to 30):', min_value=1, max_value=30, value=7)

# Assuming 'df_scaled_date_indexed' and 'test_df' are available from the previous notebook cells
# We need to load the original data to get the last window of *unscaled* data
df_original = pd.read_csv('/content/AAPL.csv')
df_original['Date'] = pd.to_datetime(df_original['Date'])
df_original.set_index('Date', inplace=True)

# Get the last window of data from the original dataframe
window_size = 20 # This should match the window size used in singleStepSampler
last_window_data_original = df_original.iloc[-window_size:].copy()


if st.sidebar.button('Generate Forecast'):
    st.subheader(f'Forecast for the next {num_days_to_forecast} days:')

    # Get the last date from the original data
    last_date = df_original.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days_to_forecast, freq='B') # 'B' for business days

    forecast_results = forecast_future_days(multivariate_lstm, scaler, target_scaler, last_window_data_original, num_days_to_forecast, window_size)

    # Add the forecast dates
    forecast_results.index = forecast_dates

    st.dataframe(forecast_results)

    st.markdown("""
    **Note:** This forecast is based on a multivariate LSTM model trained on historical data.
    Stock price prediction is inherently uncertain and this forecast should not be considered financial advice.
    The model uses the predicted 'Close' price as a placeholder for 'High', 'Low', and 'Adj Close' in subsequent predictions, and uses the average volume from the last window.
    """)
