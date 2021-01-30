import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


df = web.DataReader('AAPL', data_source = 'yahoo', start = '2012-01-01', end = '2020-12-17')
print(df)

print(df.shape)

# Visualize the price
plt.figure(figsize=(16,8))
plt.title('Closing Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price USD ($)', fontsize=18)
plt.show()

# Close column data
data = df.filter(['Close'])
dataset = data.values
trian_data_len = math.ceil(len(dataset) * .8)
print(trian_data_len)

# Scaling Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

# Creating the training data
train_data = scaled_data[0:trian_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<=60:
        print(x_train)
        print(y_train)
        print()

# converting training data into numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# reshaping the training data
x_train = np.shape(x_train, (x_train[0], x_train.shape[1], 1))
x_train.shape

# Building the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

# Compliling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epoch=1)

# Creating the testing data
test_data = scaled_data[trainn_data_len - 60, :]

x_test = []
y_test = dataset[train_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Converting the test data in numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# model predicted value
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)

# Getting the RMSE
rmse = np.sqrt(np.mean(pred - y_test)**2)
print(rmse)

# Plotting the Data
train = data[:train_data_len]
valid = data[trian_data_len:]
valid['Predictions'] = pred
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
plt.show()

# Checking the valid and predicted price
valid

# Creating the Final Model to predict the price
apple = web.DataReader('AAPL', data_source = 'yahoo', start = '2012-01-01', end = '2019-12-17')

df_1 = apple.flter(['Close'])

last_60_days = df_1[-60].values

last_60_days_scaled = scaler.transform(last_60_days)

X_test = []

X_test.append(last_60_days_scaled)

X_test = np.array(X_test)

X_test = np.reshape(X_test, (x_test.shape[0], X_test.shape[1], 1))

pred_1 = model.predict(X_test)

pred_1 = scaler.inverse_transform(pred_1)
print(pred_1)

# Actually Checking the price
apple_1 = web.DataReader('AAPL', data_source = 'yahoo', start = '2012-01-01', end = '2019-12-17')
print(apple_2['Close'])