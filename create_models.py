import keras
from keras import backend
from tensorflow.keras.layers import Bidirectional
# from keras.regularizers import l2
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Activation
from tensorflow.python.keras.callbacks import ModelCheckpoint


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


from tensorflow.keras.layers import Bidirectional
from keras.regularizers import l2
import tensorflow as tf


def create_model(x_train, y_train):
    model = Sequential()
    # model.add(LSTM(512, input_shape=(x_train.shape[1], x_train.shape[2]), kernel_initializer='uniform'))

    model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2]), kernel_initializer='uniform',
                   return_sequences=True))
    model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2]), kernel_initializer='uniform'))

    # model.add(LSTM(128, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    # model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], ))
    model.add(Activation('relu'))
    adam = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=[rmse])
    print(model.summary())

    return model


def create_model_stateful(x_train, y_train, batch_size):
    model = Sequential()
    model.add(
        LSTM(512, batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]), kernel_initializer='uniform',
             stateful=True, return_sequences=False))
    # model.add(LSTM(64, batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]), kernel_initializer='uniform',
    #                 stateful=True))

    # model.add(LSTM(256, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    # model.add(Dense(3, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], ))
    model.add(Activation('relu'))
    adam = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=[rmse])
    print(model.summary())

    return model


def create_model_bilstm(x_train, y_train):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256, return_sequences=True),
                            input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Bidirectional(LSTM(units=256, )))
    model.add(Dense(1, activation='relu'))
    # Compile model
    adam = keras.optimizers.Adam(lr=0.0001, )
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])
    print(model.summary())
    return model


def create_model_stateful_bidirectional(x_train, y_train, batch):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256, return_sequences=False),
                            batch_input_shape=(batch, x_train.shape[1], x_train.shape[2])))
    # model.add(Bidirectional(LSTM(units = 256, )))

    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='relu'))
    # Compile model
    adam = keras.optimizers.Adam(lr=0.0001, )
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])
    print(model.summary())
    return model


def run_model(x_train, y_train, x_valid=None, y_valid=None, batch_n=None, EPOCHS=None):
    from keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25, restore_best_weights=True)
    best_model = ModelCheckpoint('./best_model.h5', monitor='val_rmse', mode='min', save_best_only=True, verbose=1)

    batch_n = 64
    # model = create_model(x_train,y_train)
    model = create_model_stateful(x_train, y_train, batch_n)
    # model = create_model_bilstm(128)
    # model = create_model_stateful_bidirectional(x_train, y_train, batch_n)
    EPOCHS = 1 if EPOCHS is None else EPOCHS
    history = None
    # history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=batch_n, validation_data=(x_valid, y_valid), verbose=2, shuffle=False,
    #                   callbacks=[es, ])

    tf.keras.backend.clear_session()

    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=batch_n, validation_split=0.2,
                        verbose=2, shuffle=False, callbacks=[es, best_model])

    # history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=batch_n, verbose=2, shuffle=False,
    #                     callbacks=[es])
    return model, history




