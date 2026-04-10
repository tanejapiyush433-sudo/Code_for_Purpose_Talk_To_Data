# src/main.py
from lstm_model import run_lstm
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

from forecast import run_forecast
from anomaly import detect_anomalies
from utils import preprocess

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# LOAD DATA
# ==============================
file = glob.glob('../**/*.csv', recursive=True)[0]
df = pd.read_csv(file)

# ==============================
# PREPROCESS
# ==============================
df = preprocess(df)

# ==============================
# BASELINE (IMPORTANT)
# ==============================
df['baseline'] = df['y'].rolling(window=10).mean()

# ==============================
# TRAIN-TEST SPLIT
# ==============================
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# ==============================
# FORECAST (TRAIN ONLY)
# ==============================
forecast = run_forecast(train_df)

# ==============================
# EXTEND FORECAST FOR TEST DATA
# ==============================
future = forecast[['ds']]
forecast_full = forecast.copy()

# Predict future for test length
from prophet import Prophet

model = Prophet(interval_width=0.95, changepoint_prior_scale=0.5)
model.fit(train_df)

future = model.make_future_dataframe(periods=len(test_df), freq='s')
forecast = model.predict(future)

# ==============================
# TEST PREDICTIONS
# ==============================
forecast_test = forecast.iloc[-len(test_df):]

y_true = test_df['y'].reset_index(drop=True)
y_pred_model = forecast_test['yhat'].reset_index(drop=True)
y_pred_baseline = test_df['baseline'].reset_index(drop=True)

# ==============================
# ANOMALY DETECTION
# ==============================
df = detect_anomalies(df)

# ==============================
# EVALUATION
# ==============================
mae_model = mean_absolute_error(y_true, y_pred_model)
rmse_model = np.sqrt(mean_squared_error(y_true, y_pred_model))

mae_baseline = mean_absolute_error(y_true, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_true, y_pred_baseline))

print("\n========== TEST EVALUATION ==========")

print("\n--- Model ---")
print(f"MAE: {mae_model:.2f}")
print(f"RMSE: {rmse_model:.2f}")

print("\n--- Baseline ---")
print(f"MAE: {mae_baseline:.2f}")
print(f"RMSE: {rmse_baseline:.2f}")
# ==============================
# LSTM COMPARISON (OPTIONAL)
# ==============================
y_test_lstm, y_pred_lstm = run_lstm(df['y'])

plt.figure(figsize=(10,5))

plt.plot(y_test_lstm, label='Actual')
plt.plot(y_pred_lstm, label='LSTM Prediction')

plt.legend()
plt.title("LSTM Model Comparison")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.show()
# ==============================
# TEST VISUALIZATION
# ==============================
plt.figure(figsize=(10,5))

plt.plot(y_true.values, label='Actual (Test)')
plt.plot(y_pred_model.values, label='Model Prediction')
plt.plot(y_pred_baseline.values, label='Baseline')

plt.legend()
plt.title("Test Set Prediction vs Actual")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.show()

print("\nPipeline executed successfully")
