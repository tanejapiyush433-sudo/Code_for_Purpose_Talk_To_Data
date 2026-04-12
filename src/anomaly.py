# src/anomaly.py

import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import kurtosis


def detect_anomalies(df, contamination='auto'):
    """Detect anomalies in EEG signal data using Isolation Forest.

    Improvements over v1:
    - Dynamic contamination rate derived from signal kurtosis instead of
      a hard-coded 0.01. Spiky EEG signals (high kurtosis) warrant a higher
      contamination rate; smooth signals warrant a lower one.
    - Adds an 'anomaly_score' column (raw decision function output) alongside
      the binary 'anomaly' flag — useful for ranking severity of anomalies.
    - Prints a summary of detected anomalies and their timestamps.

    Anomaly detection is clinically relevant: sudden EEG spikes or drops can
    indicate stress episodes, cognitive overload, or artefacts — all useful
    signals for the NatWest financial wellbeing use case.

    Args:
        df (pd.DataFrame): Dataframe with a 'y' column (EEG signal values)
            and a 'ds' column (datetime).
        contamination (float or 'auto'): Expected fraction of anomalies.
            Pass 'auto' to derive from signal kurtosis (recommended).
            Pass a float (e.g. 0.05) to override manually.

    Returns:
        pd.DataFrame: Original dataframe with added columns:
            - 'anomaly' (int): 1 = anomaly detected, 0 = normal.
            - 'anomaly_score' (float): Higher = more anomalous. Useful for
              sorting and prioritising the most extreme signal events.
    """
    if contamination == 'auto':
        kurt = kurtosis(df['y'].values)
        # Clip to a safe range: 0.5% minimum, 10% maximum
        contamination = float(np.clip(0.005 + 0.002 * max(kurt, 0), 0.005, 0.10))
        print(f"[INFO] Signal kurtosis: {kurt:.3f} → contamination set to {contamination:.3f}")
    else:
        print(f"[INFO] Using manual contamination: {contamination}")

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    predictions = iso.fit_predict(df[['y']])

    # Binary flag: 1 = anomaly, 0 = normal
    df['anomaly'] = np.where(predictions == -1, 1, 0)

    # Continuous anomaly score: more negative = more anomalous
    # Negate so that higher score = more anomalous (intuitive direction)
    raw_scores = iso.decision_function(df[['y']])
    df['anomaly_score'] = -raw_scores

    # Summary report
    anomaly_count = df['anomaly'].sum()
    total = len(df)
    pct = 100 * anomaly_count / total
    print(f"[INFO] Anomalies detected: {anomaly_count} / {total} samples ({pct:.2f}%)")

    if anomaly_count > 0 and 'ds' in df.columns:
        top = df[df['anomaly'] == 1].nlargest(3, 'anomaly_score')[['ds', 'y', 'anomaly_score']]
        print("[INFO] Top 3 most severe anomalies:")
        for _, row in top.iterrows():
            print(f"       {row['ds']}  signal={row['y']:.3f}  score={row['anomaly_score']:.4f}")

    return df
