import pandas as pd
import numpy as np

from scipy.stats import norm
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import q_stat

from datetime import datetime

from numpy import concatenate
from pandas import concat
import re

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Activation

import seaborn as sns
import matplotlib.pyplot as plt

from create_models import run_model

from keras import backend
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error

import keras

from sklearn import preprocessing

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from statistics import mean
from sklearn.model_selection import TimeSeriesSplit

from keras.preprocessing.sequence import TimeseriesGenerator
from numpy import insert

import tensorflow as tf

# from utils import encode_month, encode
from dataTransformer import TimeStampFourierTransform, ColumnsSelectTransformer, TrainDataTransformer, ScaleTransformer

plt.rcParams["figure.figsize"] = (17, 8)


def calc_sde(y_true, y_pred):
    me = np.mean(y_true - y_pred)
    sse = np.sqrt((np.sum((y_pred - me) ** 2)) / y_true.shape[0])
    return sse




df = pd.read_csv("https://raw.githubusercontent.com/kpandey16/test/main/GHI/final1.csv", index_col=0, parse_dates=True)
print(df.shape)

# df["datetime"] = df.index
# # df["weekday"] = df.datetime.dt.weekday
# # df["weekday_name"] = df["datetime"].dt.day_name()
# df['Hour'] = df["datetime"].dt.hour
# df['Day'] = df["datetime"].dt.day
# df['Month'] = df["datetime"].dt.month
# df['Minute'] = df["datetime"].dt.minute
#
# df['hourfloat'] = df.Hour + df.Minute / 60.0
#
# df = encode_month(df, 'Month', 12)
# df = encode(df, 'Day', 30)
# df = encode(df, 'hourfloat', 24)
# df = encode(df, 'Wind direction', 360)
#
train_st = "2017-01-01 00:15:00"
train_end = "2019-12-31 23:45:00"

test_st = "2020-01-01 00:00:00"
test_end = "2020-01-11 23:45:00"

ds = df.loc[train_st:test_end, :]
print(ds.shape)

### 'datetime', 'Hour', 'Day', 'Month', 'Minute', 'hourfloat',

cols = ['Clear sky GHI', 'Clear sky BHI', 'Clear sky DHI',
        'Clear sky BNI', 'Temperature', 'Relative Humidity', 'Pressure',
        'Wind speed', 'Rainfall', 'Snowfall', 'Snow depth',
        'Short-wave irradiation', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
        'hourfloat_sin', 'hourfloat_cos', 'Wind direction_sin',
        'Wind direction_cos']

target = ['TOA']

final_columns = cols+target

ds = ds.loc[:,final_columns]

processing_steps = []
processing_steps.append(("FourierTransform", TimeStampFourierTransform(ds)))
processing_steps.append(("SelectColumns", ColumnsSelectTransformer(columns=final_columns) ))
processing_steps.append(("Scale", ScaleTransformer(target_col_name=target)))
processing_steps.append(("TrainDataShape", TrainDataTransformer(look_ahead=1056, look_back=1056,targetName=None)))

p1 = Pipeline(steps=processing_steps)

x, y = p1.fit_transform(ds)


# y = ds.loc[:, target]
# dd = ds.loc[:, cols]
# dd['TOA'] = y

from numpy import array





# Scaling prior to splitting
# scaler_x = MinMaxScaler(feature_range=(0.01, 0.99))
# scaler_y = MinMaxScaler(feature_range=(0.01, 0.99))
#
# cols = list(dd.columns)
# cols.remove('TOA')
# scaled_x = scaler_x.fit_transform(dd.loc[:, cols].values)
# scaled_y = scaler_y.fit_transform(dd.loc[:, 'TOA'].values.reshape(-1, 1))
#
# scaled_data = np.column_stack((scaled_x, scaled_y))
#
# look_back = 24 * 4 * 11
# look_ahead = 11 * 24 * 4
# # Build sequences
# # x_sequence, y_sequence = create_sequences(data=scaled_data, window=4, prediction_distance=28)
#
# print("Initial shape: scaled_data: {}, scaled_y: {}".format(scaled_data.shape, scaled_y.shape))
# x_sequence = create_windows(data=scaled_data, window_shape=look_back, end_id=-look_ahead)
# y_sequence = create_windows(data=scaled_y, window_shape=look_ahead, start_id=look_back)
# print("x_sequence: {} and y_sequence: {}".format(x_sequence.shape, y_sequence.shape))
#
# # test_len = int(len(x_sequence) * 0.003)
# # valid_len = int(len(x_sequence) * 0.003)
# test_len = 22
# valid_len = 22
# train_end = len(x_sequence) - (test_len + valid_len)
# x_train, y_train = x_sequence[:train_end], y_sequence[:train_end]
# x_valid, y_valid = x_sequence[train_end:train_end + valid_len], y_sequence[train_end:train_end + valid_len]
# x_test, y_test = x_sequence[train_end + valid_len:], y_sequence[train_end + valid_len:]
#
# print(x_train.shape, y_train.shape)
# print(x_valid.shape, y_valid.shape)
# print(x_test.shape, y_test.shape)
#
# #### for stateful
#
# test_len = 64
# valid_len = 64
# # train_end = len(x_sequence) - (test_len + valid_len)
# train_end = (len(x_sequence) - (len(x_sequence) % 64))
# x_train, y_train = x_sequence[:train_end], y_sequence[:train_end]
# # x_valid, y_valid = x_sequence[train_end:train_end + valid_len], y_sequence[train_end:train_end + valid_len]
# # x_test, y_test = x_sequence[train_end + valid_len:], y_sequence[train_end + valid_len:]
#
# print(x_train.shape, y_train.shape)
# # print(x_valid.shape, y_valid.shape)
# # print(x_test.shape, y_test.shape)
#
#
# tf.keras.backend.clear_session()
#
# model, history = run_model(x_train, y_train, x_valid, y_valid)
