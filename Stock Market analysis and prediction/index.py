import math
import pandas_datareader as web
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Load data and visualize
df = pd.read_csv("C:/Users/gedas/Desktop/PROJECTS/Stock Market analysis and prediction/MRF.NS.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Prepare data
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.8)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=200, epochs=1)

# Prepare testing data
test_data = scaled_data[training_data_len - 60:, :]
x_test = []

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculate RMSE
y_test = dataset[training_data_len:, :]
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print("Root Mean Squared Error:", rmse)

# Visualize predictions
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=20)
plt.ylabel('Close Price', fontsize=20)

# Use the correct column names when plotting
plt.plot(train.index, train['Close'], label='Train')
plt.plot(valid.index, valid['Close'], label='Val')
plt.plot(valid.index, valid['Predictions'], label='Predictions')

plt.legend(loc='upper left')
plt.show()


# Predict future price
stock_quote = pd.read_csv("C:/Users/gedas/Desktop/PROJECTS/Stock Market analysis and prediction/MRF.NS.csv")
stock_quote['Date'] = pd.to_datetime(stock_quote['Date']) 
new_df = stock_quote.filter(['Date','Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_list = []
X_list.append(last_60_days_scaled)
X_list = np.array(X_list)
X_test = np.reshape(X_list, (X_list.shape[0], X_list.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print("Predicted price is", pred_price)
