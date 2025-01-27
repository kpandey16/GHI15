



#### adjust for single batch
# from tensorflow.python.keras.models import load_model
from pickle import load

from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

import train
import utils
from create_models import create_model_stateful, rmse
import create_models

x_shape = (8384, 192, 21)
y_shape = (8384, 192, 1)

pred_model = create_model_stateful(time_steps=train.look_back, features=21, output_length=train.look_ahead, batch_size=1)

dependencies = {
    'rmse': rmse
}
saved_model = load_model("./best_model.h5", custom_objects=dependencies)

train_weights = saved_model.get_weights()
pred_model.set_weights(train_weights)

test_pipe = load(open('./process_pipeline.pkl', 'rb'))
scaler_y = load(open('./scaler_y.pkl', 'rb'))

mae_arr = {}
rmse_arr = {}

"""
11 days -- '2020-12-01 00:00:00':'2020-12-11 23:45:00'
7 days -- '2020-12-01 00:00:00':'2020-12-07 23:45:00'
1 day -- '2020-12-01 00:00:00':'2020-12-01 23:45:00'
"""

for i in range(1, 13):
  test_st = "2020-" + "{:02d}".format(i) + "-01 00:00:00"
  test_end = "2020-"+ "{:02d}".format(i) +"-11 23:45:00"

  y_true = train.df.loc[test_st:test_end, 'TOA']
  print("main shape: ", train.df.loc[test_st:test_end, 'TOA'].shape)

  test_past_st = train.df.index.get_loc(test_st)
  ttest2 = train.df.iloc[test_past_st - train.look_back : test_past_st, :]

  test2_scaled = test_pipe.transform(ttest2)
  # print("Pipeline processed: ", test2_scaled.shape)
  tx, ty = utils.get_X_y(X=test2_scaled, look_back=train.look_back, look_ahead=None, test_data=True)
  # print("test_X", tx.shape)

  test2_pred = pred_model.predict(tx, batch_size=1)
  test2_pred = scaler_y.inverse_transform(test2_pred)

  test2_pred = test2_pred.reshape(-1, 1)
  test2_pred[test2_pred<0]=0

  rmse = mean_squared_error(test2_pred, y_true.values)
  mae = mean_absolute_error(test2_pred,y_true.values )

  rmse_arr[i] = rmse
  mae_arr[i] = mae

print("RMSE: \n", rmse_arr)
print("MAE: \n", mae_arr)
