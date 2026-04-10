# src/anomaly.py

from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    iso = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly'] = iso.fit_predict(df[['y']])
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    
    return df
