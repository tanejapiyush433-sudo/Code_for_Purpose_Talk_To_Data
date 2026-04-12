# src/utils.py

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch


# ==============================
# EEG FREQUENCY BANDS
# ==============================
BANDS = {
    'delta': (0.5, 4.0),   # Deep sleep / unconscious
    'theta': (4.0, 8.0),   # Drowsiness / memory
    'alpha': (8.0, 13.0),  # Relaxed awareness
    'beta':  (13.0, 30.0), # Alert / stressed / focused
    'gamma': (30.0, 40.0), # High cognitive load
}


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=256.0, order=4):
    """Apply a Butterworth bandpass filter to remove noise from EEG signal.

    Filters out frequencies below 0.5 Hz (slow drift) and above 40 Hz
    (muscle artefacts, power-line noise at 50/60 Hz). This is standard
    preprocessing for clinical and research-grade EEG pipelines.

    Args:
        signal (np.ndarray): Raw 1D EEG signal values.
        lowcut (float): Low cutoff frequency in Hz. Default 0.5 Hz.
        highcut (float): High cutoff frequency in Hz. Default 40.0 Hz.
        fs (float): Sampling frequency in Hz. Default 256 Hz (common EEG rate).
        order (int): Filter order. Higher = steeper roll-off. Default 4.

    Returns:
        np.ndarray: Filtered signal, same length as input.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def compute_band_powers(signal, fs=256.0):
    """Compute relative power in each EEG frequency band using Welch's method.

    Band powers are expressed as fractions of total power (0.0–1.0).
    High beta power indicates stress or alertness; high alpha indicates
    relaxation — relevant for detecting financial vulnerability states.

    Args:
        signal (np.ndarray): Filtered 1D EEG signal values.
        fs (float): Sampling frequency in Hz. Default 256 Hz.

    Returns:
        dict: Band name → relative power, e.g. {'delta': 0.32, 'alpha': 0.18, ...}
    """
    nperseg = min(len(signal), int(fs * 2))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    total_power = np.trapz(psd, freqs)

    if total_power == 0:
        return {band: 0.0 for band in BANDS}

    band_powers = {}
    for band, (lo, hi) in BANDS.items():
        idx = np.logical_and(freqs >= lo, freqs <= hi)
        band_powers[band] = float(np.trapz(psd[idx], freqs[idx]) / total_power)

    return band_powers


def preprocess(df, channel=None, fs=256.0):
    """Preprocess raw EEG dataframe into Prophet-compatible format.

    Improvements over v1:
    - Multi-channel support: pass channel name or auto-select first numeric col.
    - Bandpass filtering (0.5–40 Hz) to remove noise before modelling.
    - NaN handling: forward-fill then drop remaining NaNs.
    - Computes and prints EEG frequency band power summary.
    - Logs which channel is selected so the user knows what was used.

    Args:
        df (pd.DataFrame): Raw EEG dataframe with one or more numeric columns.
        channel (str, optional): Column name to use as the EEG signal.
            If None, the first numeric column is used automatically.
        fs (float): Sampling frequency in Hz for filtering. Default 256 Hz.

    Returns:
        pd.DataFrame: Processed dataframe with 'ds' (datetime) and 'y' (signal) columns.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty. Check that your data file loaded correctly.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in the dataframe.")

    # --- Channel selection ---
    if channel and channel in df.columns:
        value_col = channel
    else:
        value_col = numeric_cols[0]

    print(f"[INFO] Using EEG channel : '{value_col}'")
    if len(numeric_cols) > 1:
        others = [c for c in numeric_cols if c != value_col]
        print(f"[INFO] Other channels available: {others}")
        print(f"[INFO] To use a different channel, call preprocess(df, channel='<name>')")

    # --- NaN handling ---
    signal = df[value_col].copy()
    nan_count = signal.isna().sum()
    if nan_count > 0:
        print(f"[WARN] {nan_count} NaN values found — forward-filling.")
        signal = signal.fillna(method='ffill').dropna()

    # --- Bandpass filter (removes power-line noise & drift) ---
    print(f"[INFO] Applying bandpass filter: 0.5–40 Hz @ {fs} Hz sample rate")
    filtered = bandpass_filter(signal.values, fs=fs)

    # --- Frequency band power summary ---
    band_powers = compute_band_powers(filtered, fs=fs)
    print("\n[INFO] EEG Band Power Summary:")
    for band, power in band_powers.items():
        bar = '█' * int(power * 40)
        print(f"       {band:<6} {power:.3f}  {bar}")
    dominant = max(band_powers, key=band_powers.get)
    print(f"[INFO] Dominant band: {dominant.upper()} "
          f"({'stressed/alert' if dominant == 'beta' else 'relaxed' if dominant == 'alpha' else dominant})")

    # --- Build output dataframe ---
    out = pd.DataFrame()
    out['ds'] = pd.date_range(start='2024-01-01', periods=len(filtered), freq='s')
    out['y'] = filtered

    return out[['ds', 'y']]
