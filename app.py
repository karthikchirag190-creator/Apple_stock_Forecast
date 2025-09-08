import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime

# Load the trained model, scalers, and imputer
try:
    multivariate_lstm = joblib.load('multivariate_lstm_model.joblib')
    feature_scaler = joblib.load('feature_scaler.joblib')
    target_scaler = joblib.load('target_scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler files not found. Please make sure 'multivariate_lstm_model.joblib', 'feature_scaler.joblib', and 'target_scaler.joblib' are in the same directory.")
    st.stop()

# Set the title of the app
st.title('Apple Stock Price Prediction')

# Option for user input: CSV upload or manual input
input_option = st.radio("Choose input method:", ('Upload CSV', 'Manual Input'))

# Define the features to be used for the model
features_to_use = ['High', 'Low', 'Close', 'Adj Close', 'Volume']
window_size = 20 # Define window_size here

if input_option == 'Upload CSV':
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        # Ensure the Date column is in datetime format
        input_df['Date'] = pd.to_datetime(input_df['Date'])
        input_df.set_index('Date', inplace=True)

        # Display the uploaded data
        st.write("Uploaded Data:")
        st.write(input_df.head())

        # Preprocess the input data
        try:
            # Apply the same preprocessing steps as during training
            # Select only the columns used for training
            input_features = input_df[features_to_use]

            # Scale the input features
            input_scaled = feature_scaler.transform(input_features)
            input_scaled_df = pd.DataFrame(input_scaled, columns=features_to_use, index=input_df.index)

            # Prepare data for prediction (using the last 20 days as the window)

            if len(input_scaled_df) < window_size:
                st.warning(f"Please provide at least {window_size} days of data for prediction.")
            else:
                X_predict = []
                for i in range(len(input_scaled_df) - window_size + 1):
                    X_predict.append(input_scaled_df.iloc[i:i+window_size].values)
                X_predict = np.array(X_predict)

                # Make predictions
                predicted_scaled = multivariate_lstm.predict(X_predict)

                # Inverse transform the predictions to get actual price values
                # The model predicts two outputs, we assume the second one is 'Close'
                dummy_array = np.zeros((predicted_scaled.shape[0], target_scaler.n_features_in_))
                dummy_array[:, 1] = predicted_scaled[:, 1] # Assign predicted 'Close' to the correct column
                predicted_actual = target_scaler.inverse_transform(dummy_array)[:, 1]

                # Create a DataFrame with predictions and corresponding dates
                prediction_dates = input_df.index[window_size-1:]
                predictions_df = pd.DataFrame({'Date': prediction_dates, 'Predicted_Close': predicted_actual})
                predictions_df.set_index('Date', inplace=True)


                st.write("Predicted Closing Prices:")
                st.write(predictions_df)

                # Plot the predictions
                fig = px.line(predictions_df, y='Predicted_Close', title='Predicted Closing Prices')
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

elif input_option == 'Manual Input':
    st.write("Enter the stock data for the day you want to predict:")

    input_data = {}
    # Use only the selected features for manual input
    manual_input_columns = features_to_use

    date_input = st.date_input("Date")
    for col in manual_input_columns:
        input_data[col] = st.number_input(f'{col}', value=0.0)

    if st.button('Predict'):
        try:
            # Create a DataFrame from manual input
            manual_input_df = pd.DataFrame([input_data], index=[pd.to_datetime(date_input)])

            # Ensure data types are correct
            manual_input_df = manual_input_df.astype(float)

            # We need a window of 20 days for prediction.
            # Since the user only provides one day, we need to create a 20-day sequence
            # for prediction.
            # This part assumes you have a way to get the preceding 19 days of data.
            # For a real-world application, you would need a data source to fetch this.
            # For this example, we will just create a dummy sequence for demonstration.

            # Create a dummy 20-day sequence using the manual input data
            # In a real scenario, replace this with actual preceding data
            dummy_sequence = np.zeros((window_size, len(features_to_use)))
            dummy_sequence[-1, :] = manual_input_df.values # Place the manual input as the last day

            # Scale the dummy sequence
            scaled_sequence = feature_scaler.transform(dummy_sequence)

            # Prepare data for prediction
            X_predict_manual = np.array([scaled_sequence]) # Reshape for the model

            # Make predictions
            predicted_scaled_manual = multivariate_lstm.predict(X_predict_manual)

            # Inverse transform the predictions
            dummy_array_manual = np.zeros((predicted_scaled_manual.shape[0], target_scaler.n_features_in_))
            dummy_array_manual[:, 1] = predicted_scaled_manual[:, 1]
            predicted_actual_manual = target_scaler.inverse_transform(dummy_array_manual)[:, 1]

            st.write("Predicted Closing Price for the next day:")
            st.write(predicted_actual_manual[0])

        except Exception as e:
             st.error(f"An error occurred during prediction: {e}")
