import pandas as pd
from sklearn.pipeline import Pipeline

from create_models import run_model
from dataTransformer import TimeStampFourierTransform, ColumnsSelectTransformer, ScaleTransformer, PreProcessData

from pickle import dump, load
import Configs
import utils

pd.set_option('mode.chained_assignment', None)

# from utils import encode_month, encode
# from sklearn.preprocessing import FunctionTransformer


df = pd.read_csv("./new_file.csv", index_col=0, parse_dates=True)
print(df.shape)

train_st = "2019-10-01 00:15:00"
train_end = "2019-12-31 23:45:00"

test_st = "2020-01-01 00:00:00"
test_end = "2020-01-11 23:45:00"

trainset = df.loc[train_st:train_end, :]
testset = df.loc[test_st:test_end, :]
print("train set: ", trainset.shape)
print("test set: ", testset.shape)

final_columns = Configs.cols + Configs.target

processing_steps = []
processing_steps.append(("PreProcess", PreProcessData()))
processing_steps.append(("FourierTransform", TimeStampFourierTransform(colnames=None, max_val=None)))
processing_steps.append(("SelectColumns", ColumnsSelectTransformer(columns=final_columns, )))
processing_steps.append(("Scale", ScaleTransformer(target_col_name=Configs.target[0])))

# processing_steps.append(("TrainDataShape", TrainDataTransformer(look_ahead=2, look_back=2, targetName=None)))

p1 = Pipeline(steps=processing_steps)

scaled_data = p1.fit_transform(trainset)
print("Scaled data: ", scaled_data.shape)

# dump(p1, open("./process_pipeline.pkl", "wb"))


look_back = 4 * 24 * 2
look_ahead = 4 * 24 * 2
x, y = utils.get_X_y(X=scaled_data, look_back=look_back, look_ahead=look_ahead)

test_len = 64
valid_len = 64
train_end = len(x) - (test_len + valid_len)
train_end = len(x) - (valid_len)
train_init = train_end % 64
# train_end = (len(x_sequence)-(len(x_sequence)))
# train_end = len(x_sequence)
x_train, y_train = x[train_init:train_end], y[train_init:train_end]
x_valid, y_valid = x[train_end:], y[train_end:]
# x_valid, y_valid = x[train_end:train_end + valid_len], x[train_end:train_end + valid_len]
# x_test, y_test = x[train_end + valid_len:], x[train_end + valid_len:]

print("===================================================")
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
# print(x_test.shape, y_test.shape)

print("==============================================")

# run_model(x_train, y_train, y_valid=y_valid, x_valid=x_valid, batch_n=64, EPOCHS=2)

# test_pipe = load(open('./process_pipeline.pkl', 'rb'))
# test_scaled = test_pipe.fit_transform(testset)
#
# tx, ty = utils.get_X_y(X=test_scaled, look_back=look_back, look_ahead=look_back, test_data=True)
# print(tx.shape)
