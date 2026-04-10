# src/main.py

import pandas as pd
import glob

from forecast import run_forecast
from anomaly import detect_anomalies
from utils import preprocess

# Load data
file = glob.glob('../**/*.csv', recursive=True)[0]
df = pd.read_csv(file)

# Process
df = preprocess(df)

# ==============================
# TRAIN-TEST SPLIT
# ==============================
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Forecast
forecast = run_forecast(df)

# Anomaly
df = detect_anomalies(df)

print("Pipeline executed successfully")
