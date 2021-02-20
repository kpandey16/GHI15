



#### adjust for single batch
# from tensorflow.python.keras.models import load_model
from pickle import load

from keras.models import load_model
from sklearn.metrics import mean_absolute_error

import train
import utils
from create_models import create_model_stateful, rmse

x = train.x
y = train.y

pred_model = create_model_stateful(x, y, 1)

dependencies = {
    'rmse': rmse
}
saved_model = load_model("./best_model.h5", custom_objects=dependencies)

train_weights = saved_model.get_weights()
pred_model.set_weights(train_weights)

# def transform_test_data(df, test_st, test_end):
#   test_past_st = df.index.get_loc(test_st)
#   test2 = df.iloc[test_past_st-1056:test_past_st]
#
#   scaled_x = scaler_x.transform(test2.loc[:, cols].values)
#   scaled_y = scaler_y.transform(test2.loc[:, 'TOA'].values.reshape(-1,1))
#
#   test2_scaled = np.column_stack((scaled_x, scaled_y))
#
#   test2_scaled = test2_scaled.reshape(1,1056,21)
#   # print(test2_scaled.shape)
#   return test2_scaled, test2

# test_st = "2020-03-03 00:00:00"
# test_end = "2020-03-14 23:45:00"

# testset = train.df.loc[test_st:test_end, :]

# test_past_st = train.df.index.get_loc(test_st)
# test2 = train.df.iloc[test_past_st - train.look_back : test_past_st, :]

test_pipe = load(open('./process_pipeline.pkl', 'rb'))
scaler_y = load(open('./scaler_y.pkl', 'rb'))
# test_scaled = test_pipe.fit_transform(test2)

# tx, ty = utils.get_X_y(X=test_scaled, look_back=train.look_back, look_ahead=train.look_back, test_data=True)
# print(tx.shape)


err =[]
for i in range(1, 13):
  test_st = "2020-" + "{:02d}".format(i) + "-01 00:00:00"
  test_end = "2020-"+ "{:02d}".format(i) +"-02 23:45:00"

  y_true = train.df.loc[test_st:test_end, 'TOA']

  test_past_st = train.df.index.get_loc(test_st)
  ttest2 = train.df.iloc[test_past_st - train.look_back : test_past_st, :]

  test2_scaled = test_pipe.fit_transform(ttest2)
  tx, ty = utils.get_X_y(X=test2_scaled, look_back=train.look_back, look_ahead=train.look_back, test_data=True)
  print("test_X", tx.shape)

  test2_pred = pred_model.predict(tx, batch_size=1)
  test2_pred = scaler_y.inverse_transform(test2_pred)

  test2_pred = test2_pred.reshape(-1, 1)
  test2_pred[test2_pred<0]=0

  mae = mean_absolute_error(test2_pred,y_true.values )
  print(i, ".  MAE: ", mae)
  err.append(mae)
