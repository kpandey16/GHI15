import numpy as np


# def encode(data, col, max_val):
#     data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
#     data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
#     return data
#
#
# def encode_month(data, col, max_val):
#     data[col + '_sin'] = np.sin(2 * np.pi * (data[col] - 1) / max_val)
#     data[col + '_cos'] = np.cos(2 * np.pi * (data[col] - 1) / max_val)
#     return data

def create_sequences(data, window=15, step=1, prediction_distance=15):
    x = []
    y = []

    for i in range(0, len(data) - window - prediction_distance, step):
        x.append(data[i:i + window])
        y.append(data[i + window + prediction_distance][-1])

    x, y = np.asarray(x), np.asarray(y)
    print("in create_seq: ", x.shape, y.shape)
    return x, y


def create_windows(data, window_shape, step=1, start_id=None, end_id=None):
    data = np.asarray(data)
    data = data.reshape(-1, 1) if np.prod(data.shape) == max(data.shape) else data
    # print("init data: ", data.shape, "window: ", window_shape)
    start_id = 0 if start_id is None else start_id
    end_id = data.shape[0] if end_id is None else end_id
    # print("start: {}, end: {}".format(start_id, end_id))
    data = data[int(start_id):int(end_id), :]
    # print("new data: ", data.shape)
    window_shape = (int(window_shape), data.shape[-1])
    step = (int(step),) * data.ndim
    slices = tuple(slice(None, None, st) for st in step)
    # print("window shape: {}, slices: {}, step: {}".format(window_shape, slices, step))
    indexing_strides = data[slices].strides
    # print(indexing_strides)
    win_indices_shape = ((np.array(data.shape) - window_shape) // step) + 1
    # print("winindices: ", win_indices_shape)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(data.strides))
    # print("New shape: ", new_shape)
    # print("Strides: ", strides)
    window_data = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=strides)

    return np.squeeze(window_data, 1)

def get_X_y(X, look_back=1, look_ahead=1, test_data=False):
    scaled_data = X
    scaled_y = X[:, -1]
    if test_data:
        look_ahead=None
        x_sequence = create_windows(data=scaled_data, window_shape=look_back, end_id=look_ahead)
        y_sequence=None
    else:
        x_sequence = create_windows(data=scaled_data, window_shape=look_back, end_id=-look_ahead)
        y_sequence = create_windows(data=scaled_y, window_shape=look_ahead, start_id=look_back)

    return x_sequence, y_sequence