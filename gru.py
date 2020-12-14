from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def init_tensorflow():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

from sklearn.preprocessing import MinMaxScaler

def normalize_data(df):
    min_max_scaler = MinMaxScaler()
    df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1))
    df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1))
    df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1))
    df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return df


valid_set_size_percentage = 10
test_set_size_percentage = 10

def load_data(stock, seq_len):
    data_raw = stock.values
    data = []
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])
    data = np.array(data);
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]


def reader_csv_file(filename):
    dataset = read_csv(filename, index_col = 0)
    df_stock = dataset.dropna()
    df_stock = df_stock[['Open', 'High', 'Low', 'Close']]
    print('-------')
    print('File Shape', df_stock.shape)
    print(df_stock)
    print('-------')
    print('--- Normalized Data ---')
    df_stock = normalize_data(df_stock)
    print(df_stock)

    seq_len = 20 # taken sequence length as 20

    return  load_data(df_stock, seq_len)

import datetime
def GRU_KERAS(x_train, y_train, x_test, y_test):
    m, n, l = x_train.shape
    # print(x)
    model = Sequential()
    model.add(GRU(int(m / 2), input_shape = (n, l), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x_train, y_train, batch_size=52, epochs=10, callbacks=[tensorboard_callback])
    y_pred = model.predict(x_test)
    print(y_pred)
    print(y_pred.shape)
    print(x_test)
    comp = pd.DataFrame({'Column1':y_test[:,3],'Column2':y_pred[:,3]})
    comp1 = pd.DataFrame({'Column3':y_test[:,2],'Column4':y_pred[:,2]})
    plt.figure(figsize=(10,5))
    plt.plot(comp['Column1'], color='blue', label='Target')
    plt.plot(comp['Column2'], color='black', label='Prediction')
    plt.plot(comp1['Column3'], color='red', label='Target')
    plt.plot(comp1['Column4'], color='orange', label='Prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    init_tensorflow()
    x_train, y_train, x_valid, y_valid, x_test, y_test = reader_csv_file('stocks.csv')
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_valid.shape)
    # print(y_valid.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    GRU_KERAS(x_train, y_train, x_test, y_test)
    # print(np.array(x_train))

