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
            # Select only the columns used for training (excluding 'Date')
            input_features = input_df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

            # Scale the input features
            input_scaled = feature_scaler.transform(input_features)
            input_scaled_df = pd.DataFrame(input_scaled, columns=input_features.columns, index=input_df.index)

            # Prepare data for prediction (using the last 20 days as the window)
            window_size = 20
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
                # We need to create a dummy array with the same shape as the training target data
                # before inverse transforming the 'Close' column.
                dummy_array = np.zeros((predicted_scaled.shape[0], 2))
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
    st.write("Enter the stock data for the last 20 days:")

    input_data = {}
    columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    dates = []
    for i in range(20):
        st.subheader(f"Day {i+1}")
        date_input = st.date_input(f"Date for Day {i+1}", datetime.date.today() - datetime.timedelta(days=19-i))
        dates.append(date_input)
        for col in columns:
            input_data[f'{col}_{i+1}'] = st.number_input(f'{col} for Day {i+1}', value=0.0)

    if st.button('Predict'):
        try:
            # Create a DataFrame from manual input
            manual_input_df = pd.DataFrame(index=pd.to_datetime(dates), columns=columns)
            for i in range(20):
                for col in columns:
                    manual_input_df.loc[pd.to_datetime(dates[i]), col] = input_data[f'{col}_{i+1}']

            # Ensure data types are correct
            manual_input_df = manual_input_df.astype(float)

            # Scale the manual input data
            manual_input_scaled = feature_scaler.transform(manual_input_df)
            manual_input_scaled_df = pd.DataFrame(manual_input_scaled, columns=columns, index=manual_input_df.index)

            # Prepare data for prediction
            X_predict_manual = np.array([manual_input_scaled_df.values]) # Reshape for the model

            # Make predictions
            predicted_scaled_manual = multivariate_lstm.predict(X_predict_manual)

            # Inverse transform the predictions
            dummy_array_manual = np.zeros((predicted_scaled_manual.shape[0], 2))
            dummy_array_manual[:, 1] = predicted_scaled_manual[:, 1]
            predicted_actual_manual = target_scaler.inverse_transform(dummy_array_manual)[:, 1]


            st.write("Predicted Closing Price for the next day:")
            st.write(predicted_actual_manual[0])

        except Exception as e:
             st.error(f"An error occurred during prediction: {e}")
