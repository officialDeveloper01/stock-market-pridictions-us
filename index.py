# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()
stock = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'

# Fetch data using pandas_datareader with Yahoo Finance
df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
print(df.head())


# Fetch data
df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)

# Step 2: Preprocess the Data
df = df[['Close']]  # Use only the 'Close' price
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# Create a dataset with time steps (e.g., 60 days)
time_step = 60
X = []
y = []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i - time_step:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

# Step 3: Build the LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)  # Predict one value (next stock price)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the Model
model.fit(X, y, batch_size=32, epochs=20)

# Step 5: Test the Model
test_start = '2023-01-02'
test_end = '2023-12-01'

test_data = pdr.get_data_yahoo(stock, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

# Prepare test inputs
total_data = pd.concat((df['Close'], test_data['Close']), axis=0)
inputs = total_data[len(total_data) - len(test_data) - time_step:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(time_step, len(inputs)):
    X_test.append(inputs[i - time_step:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Step 6: Visualize Results
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label=f'Actual {stock} Price')
plt.plot(predictions, color='red', label=f'Predicted {stock} Price')
plt.title(f'{stock} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
