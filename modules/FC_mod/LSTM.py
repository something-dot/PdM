import pandas as pd
import numpy as np

from .preprocessing import flatten


def temporalize(lookback, input_X, input_y):
    X = []
    y = []
    for i in range(len(input_X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(input_X[[(i+j+1)], :])
        X.append(t)
        y.append(input_y[i+lookback+1])
    return X, y


def scale(X, scaler):
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])

    return X


def test_train_split_standardise_lstm(df, lookback = 50 ):
    input_X = df.loc[:, df.columns != 'y'].values  # converts the df to a numpy array
    input_y = df['y'].values
    n_features = input_X.shape[1]  # number of features
    # Temporalize the data
    X, y = temporalize(lookback,input_X = input_X,input_y = input_y)
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=DATA_SPLIT_PCT, random_state=SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=DATA_SPLIT_PCT, random_state=SEED)
    X_train_y0 = X_train[y_train==0]
    X_train_y1 = X_train[y_train==1]
    X_valid_y0 = X_valid[y_valid==0]
    X_valid_y1 = X_valid[y_valid==1]
    X_train = X_train.reshape(X_train.shape[0], lookback, n_features)
    X_train_y0 = X_train_y0.reshape(X_train_y0.shape[0], lookback, n_features)
    X_train_y1 = X_train_y1.reshape(X_train_y1.shape[0], lookback, n_features)
    X_valid = X_valid.reshape(X_valid.shape[0], lookback, n_features)
    X_valid_y0 = X_valid_y0.reshape(X_valid_y0.shape[0], lookback, n_features)
    X_valid_y1 = X_valid_y1.reshape(X_valid_y1.shape[0], lookback, n_features)
    X_test = X_test.reshape(X_test.shape[0], lookback, n_features)
    scaler = StandardScaler().fit(flatten(X_train_y0))
    X_train_y0_scaled = scale(X_train_y0, scaler)
    a = flatten(X_train_y0_scaled)
    X_valid_scaled = scale(X_valid, scaler)
    X_valid_y0_scaled = scale(X_valid_y0, scaler)
    X_test_scaled = scale(X_test, scaler)
    timesteps =  X_train_y0_scaled.shape[1] # equal to the lookback
    n_features =  X_train_y0_scaled.shape[2] # 59
    return X_train_y0_scaled,X_valid_y0_scaled,X_test_scaled,timesteps,n_features,X_valid_scaled,X_test_scaled,y_valid,y_test


def lstm_autoencoder_model(path, X_train_y0_scaled =0,X_valid_y0_scaled=0,nb_epoch = 200,batch_size = 128,timesteps = 0,n_features= 0,learning_rate = 0.01,activation = "relu",optimiser = "Adam",path = "/Users/nrkvarma/Downloads/FW%3a_%5bE%5d_Re%3a_Files_for_analysis_work/lstm_autoencoder_classifier.h5"):
    lstm_autoencoder = Sequential()
    # Encoder
    lstm_autoencoder.add(LSTM(32, activation= activation, input_shape=(timesteps, n_features), return_sequences=True))
    lstm_autoencoder.add(LSTM(16, activation= activation, return_sequences=False))
    lstm_autoencoder.add(RepeatVector(timesteps))
    # Decoder
    lstm_autoencoder.add(LSTM(16, activation= activation, return_sequences=True))
    lstm_autoencoder.add(LSTM(332, activation= activation, return_sequences=True))
    lstm_autoencoder.add(TimeDistributed(Dense(n_features)))
    lstm_autoencoder.compile(loss='mse', optimizer=optimiser)

    cp = ModelCheckpoint(filepath=path,
                                   save_best_only=True,
                                   verbose=0)

    lstm_autoencoder.fit(X_train_y0_scaled, X_train_y0_scaled,
                                                    epochs=nb_epoch,
                                                    batch_size=batch_size,
                                                    validation_data=(X_valid_y0_scaled, X_valid_y0_scaled),
                                                    verbose=2,callbacks=[cp])

    return lstm_autoencoder


if __name__ == '__main__':
    # Read in data
    df = pd.read_csv(sys.argv[1])
    # Remove time column, and the categorical columns
    df = df.drop(['time', 'x28', 'x61'], axis=1)