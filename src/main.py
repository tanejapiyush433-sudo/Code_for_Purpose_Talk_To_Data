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

# Forecast
forecast = run_forecast(df)

# Anomaly
df = detect_anomalies(df)

print("Pipeline executed successfully")
