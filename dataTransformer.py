import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from utils import create_windows

from pickle import dump, load


class PreProcessData(BaseEstimator, TransformerMixin):
    def __init__(self, feature=None):
        self.feature = feature

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df["datetime"] = df.index
        df['Hour'] = df["datetime"].dt.hour
        df['Day'] = df["datetime"].dt.day
        df['Month'] = df["datetime"].dt.month
        df['Minute'] = df["datetime"].dt.minute

        df['hourfloat'] = df.Hour + df.Minute / 60.0

        return df




class TimeStampFourierTransform(BaseEstimator, TransformerMixin):

    def __init__(self, colnames, max_val=None):
        self.colnames = colnames
        self.max_val = max_val

    def fit(self, df, y=None):
        return self;

    def transform(self, df, y=None):
        df = self.encode_month(df, 'Month', 12)
        df = self.encode_others(df, 'Day', 30)
        df = self.encode_others(df, 'hourfloat', 24)
        df = self.encode_others(df, 'Wind direction', 360)

        return df

    def encode_others(self, data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
        return data

    def encode_month(self, data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * (data[col] - 1) / max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * (data[col] - 1) / max_val)
        return data


class ColumnsSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None):
        return self;

    def transform(self, df, y=None):

        df = df.loc[:, self.columns]
        return df



class ScaleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col_name):
        self.target_col_name = target_col_name

    def fit(self, df, y=None):
        return self;

    def transform(self, df, y=None):
        scaler_x = MinMaxScaler(feature_range=(0.01, 0.99))
        scaler_y = MinMaxScaler(feature_range=(0.01, 0.99))

        cols = list(df.columns)
        cols.remove(self.target_col_name)

        scaled_x = scaler_x.fit_transform(df.loc[:, cols].values)
        scaled_y = scaler_y.fit_transform(df.loc[:, self.target_col_name].values.reshape(-1, 1))

        scaled_data = np.column_stack((scaled_x, scaled_y))

        dump(scaler_x, open("./scaler_x.pkl", "wb"))
        dump(scaler_y, open("./scaler_y.pkl", "wb"))

        # return scaled_x, scaled_y
        return scaled_data


class TrainDataTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, targetName, look_back, look_ahead):
        print("init")
        self.targetName = targetName
        self.look_back = look_back
        self.look_ahead = look_ahead

    def fit(self, df, y=None):
        return self

    def transform(self, df, scaled_y=None):
        scaled_data = df
        scaled_y = df[:,-1]
        # scaled_data = np.column_stack((scaled_x, scaled_y))

        x_sequence = create_windows(data=scaled_data, window_shape=self.look_back, end_id=-self.look_ahead)
        y_sequence = create_windows(data=scaled_y, window_shape=self.look_ahead, start_id=self.look_back)

        return x_sequence, y_sequence




