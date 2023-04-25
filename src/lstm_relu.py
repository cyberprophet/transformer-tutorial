import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import dotenv
import os


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)

    return numerator / (denominator + 1e-9)


dotenv.load_dotenv()

base_url = os.getenv('BASE')
chart_route = os.getenv('CHART')
option_route = os.getenv('DAY')

code = '005930'
period = 128

res = requests.get(
    f'https://{base_url}/{chart_route}/{option_route}?code={code}&period={period}')

df = pd.json_normalize(res.json()['chart'])

dfx = df[['current', 'high', 'low', 'start', 'volume']]

dfx = MinMaxScaler(dfx)

dfy = dfx[['current']]
dfx = dfx[['high', 'low', 'start', 'volume']]

x = dfx.values.tolist()
y = dfy.values.tolist()

window_size = 10

data_x = []
data_y = []

for i in range(len(y) - window_size):
    _x = x[i: i + window_size]
    _y = y[i + window_size]

    data_x.append(_x)
    data_y.append(_y)

train_size = int(len(data_y) * 0.7)

train_X = np.array(data_x[0: train_size])
train_y = np.array(data_y[0: train_size])

test_size = len(data_y) - train_size

test_X = np.array(data_x[train_size: len(data_x)])
test_y = np.array(data_y[train_size: len(data_y)])


model = tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(units=20, activation='relu',
          return_sequences=True, input_shape=(10, 4)))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.LSTM(units=20, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(units=1))

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_X, train_y, epochs=70, batch_size=30)

pred_y = model.predict(test_X)

plt.figure()
plt.plot(test_y, color='red', label='real SEC stock price')
plt.plot(pred_y, color='blue', label='predicted SEC stock price')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()

plt.savefig(f'/mnt/g/Blog/{code}.png')
