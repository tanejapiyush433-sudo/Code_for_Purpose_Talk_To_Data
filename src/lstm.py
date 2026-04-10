import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def run_lstm(series):
    data = series.values.reshape(-1,1)

    # normalize
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # create sequences
    X, y = [], []
    for i in range(10, len(data)):
        X.append(data[i-10:i])
        y.append(data[i])

    X, y = np.array(X), np.array(y)

    # split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1],1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    preds = model.predict(X_test)

    # inverse transform
    preds = scaler.inverse_transform(preds)
    y_test = scaler.inverse_transform(y_test)

    return y_test.flatten(), preds.flatten()
