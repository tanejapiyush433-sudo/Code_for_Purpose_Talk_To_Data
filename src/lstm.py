# src/lstm.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping


# Longer window captures slower EEG rhythms (alpha ~10 Hz, theta ~6 Hz)
# At 256 Hz sample rate, 30 samples = ~117 ms of signal context
WINDOW_SIZE = 30   # was 10


def build_sequences(data, window_size):
    """Create sliding window input/output sequences for LSTM training.

    Args:
        data (np.ndarray): Normalised 2D array of shape (n_samples, 1).
        window_size (int): Number of past time steps used to predict the next.

    Returns:
        tuple: (X, y) numpy arrays where X has shape (n, window, 1).
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def run_lstm(series, window_size=WINDOW_SIZE):
    """Train an LSTM model on EEG signal data and return predictions vs actuals.

    Improvements over v1:
    - Window increased from 10 → 30 samples to capture slower EEG rhythms.
    - Added a second LSTM layer (stacked) for better temporal representation.
    - Added Dropout (0.2) after each LSTM layer to reduce overfitting.
    - Increased epochs from 5 → 50 with EarlyStopping (patience=7) so
      training stops when validation loss plateaus rather than at a fixed count.
    - Uses validation_split=0.1 to monitor generalisation during training.
    - Prints MAE and RMSE for the LSTM on the test set for easy comparison.

    Args:
        series (pd.Series): 1D time series of EEG signal values.
        window_size (int): Sliding window length. Default 30.

    Returns:
        tuple: (y_test, y_pred) — both 1D numpy arrays in original signal scale.
    """
    if len(series) < window_size * 3:
        raise ValueError(
            f"Series too short ({len(series)} samples) for window_size={window_size}. "
            f"Need at least {window_size * 3} samples."
        )

    data = series.values.reshape(-1, 1)

    # Normalise to [0, 1] for stable LSTM training
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Build sequences
    X, y = build_sequences(data, window_size)

    # 80/20 train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Build stacked LSTM model ---
    model = Sequential([
        LSTM(64, input_shape=(window_size, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')

    # --- Train with early stopping ---
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,          # was 5 — EarlyStopping handles the actual cutoff
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )

    epochs_run = len(history.history['loss'])
    print(f"[INFO] LSTM trained for {epochs_run} epochs (early stopping patience=7)")

    # --- Predict and inverse transform ---
    preds = model.predict(X_test, verbose=0)
    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test)

    # --- Evaluate ---
    mae = mean_absolute_error(y_test_inv, preds_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
    print(f"[INFO] LSTM Test MAE:  {mae:.4f}")
    print(f"[INFO] LSTM Test RMSE: {rmse:.4f}")

    return y_test_inv.flatten(), preds_inv.flatten()
