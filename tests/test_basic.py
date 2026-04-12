# tests/test_basic.py

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import preprocess, bandpass_filter, compute_band_powers, BANDS
from anomaly import detect_anomalies
from forecast import run_forecast
from lstm import run_lstm, build_sequences


# ==============================
# utils.py — preprocess()
# ==============================

def test_preprocess_output_columns():
    """preprocess() must return a dataframe with exactly 'ds' and 'y' columns."""
    raw = pd.DataFrame({'signal': np.random.randn(512)})
    result = preprocess(raw)
    assert list(result.columns) == ['ds', 'y'], "Output must have 'ds' and 'y' columns"


def test_preprocess_length_preserved():
    """preprocess() must not drop any rows for clean input data."""
    raw = pd.DataFrame({'signal': np.random.randn(512)})
    result = preprocess(raw)
    assert len(result) == 512, "Row count must be preserved after preprocessing"


def test_preprocess_empty_dataframe_raises():
    """preprocess() must raise ValueError on an empty dataframe, not crash silently."""
    with pytest.raises(ValueError, match="empty"):
        preprocess(pd.DataFrame())


def test_preprocess_no_numeric_columns_raises():
    """preprocess() must raise ValueError when no numeric columns are present."""
    raw = pd.DataFrame({'label': ['a', 'b', 'c']})
    with pytest.raises(ValueError, match="No numeric"):
        preprocess(raw)


def test_preprocess_channel_selection():
    """preprocess() must use the specified channel when provided."""
    raw = pd.DataFrame({
        'Fp1': np.ones(512) * 10.0,
        'Fp2': np.ones(512) * 99.0,
    })
    result = preprocess(raw, channel='Fp2')
    # After filtering, the mean should be close to 99 (not 10)
    assert result['y'].mean() > 50, "Should have selected the Fp2 channel (mean ~99)"


def test_preprocess_datetime_index_correct():
    """preprocess() datetime index must start at 2024-01-01 and increment by 1 second."""
    raw = pd.DataFrame({'signal': np.random.randn(10)})
    result = preprocess(raw)
    assert str(result['ds'].iloc[0].date()) == '2024-01-01'
    delta = (result['ds'].iloc[1] - result['ds'].iloc[0]).total_seconds()
    assert delta == 1.0, "Datetime index must have 1-second intervals"


# ==============================
# utils.py — bandpass_filter()
# ==============================

def test_bandpass_filter_preserves_length():
    """bandpass_filter() output must be the same length as the input signal."""
    signal = np.random.randn(1024)
    filtered = bandpass_filter(signal, fs=256.0)
    assert len(filtered) == len(signal), "Filter must not change signal length"


def test_bandpass_filter_reduces_high_freq_noise():
    """bandpass_filter() must attenuate a 100 Hz signal significantly."""
    t = np.linspace(0, 1, 256)
    noise_100hz = np.sin(2 * np.pi * 100 * t)   # 100 Hz — above cutoff
    filtered = bandpass_filter(noise_100hz, highcut=40.0, fs=256.0)
    # The filtered amplitude must be much smaller than the original
    assert np.std(filtered) < 0.1 * np.std(noise_100hz), \
        "100 Hz noise should be strongly attenuated by the 40 Hz lowpass cutoff"


# ==============================
# utils.py — compute_band_powers()
# ==============================

def test_band_powers_returns_all_bands():
    """compute_band_powers() must return a key for every band in BANDS."""
    signal = np.random.randn(2048)
    powers = compute_band_powers(signal, fs=256.0)
    assert set(powers.keys()) == set(BANDS.keys()), \
        f"Expected bands: {set(BANDS.keys())}, got: {set(powers.keys())}"


def test_band_powers_sum_approximately_one():
    """Band relative powers must sum to approximately 1.0 (within 5% tolerance)."""
    signal = np.random.randn(2048)
    powers = compute_band_powers(signal, fs=256.0)
    total = sum(powers.values())
    assert abs(total - 1.0) < 0.05, \
        f"Band powers must sum to ~1.0, got {total:.4f}"


def test_band_powers_all_non_negative():
    """All band power values must be non-negative."""
    signal = np.random.randn(2048)
    powers = compute_band_powers(signal, fs=256.0)
    for band, power in powers.items():
        assert power >= 0, f"Band '{band}' has negative power: {power}"


# ==============================
# anomaly.py — detect_anomalies()
# ==============================

def test_detect_anomalies_adds_columns():
    """detect_anomalies() must add both 'anomaly' and 'anomaly_score' columns."""
    df = pd.DataFrame({'y': np.random.randn(300)})
    result = detect_anomalies(df)
    assert 'anomaly' in result.columns,       "'anomaly' column must be present"
    assert 'anomaly_score' in result.columns, "'anomaly_score' column must be present"


def test_detect_anomalies_binary_flag():
    """'anomaly' column must contain only 0 (normal) or 1 (anomaly)."""
    df = pd.DataFrame({'y': np.random.randn(300)})
    result = detect_anomalies(df)
    unique = set(result['anomaly'].unique())
    assert unique.issubset({0, 1}), f"Anomaly flag must be 0 or 1, got: {unique}"


def test_detect_anomalies_detects_obvious_spike():
    """detect_anomalies() must flag a clear outlier spike as an anomaly."""
    np.random.seed(0)
    normal = np.random.randn(299) * 1.0   # low amplitude noise
    spike = np.array([500.0])             # massive spike
    df = pd.DataFrame({'y': np.concatenate([normal, spike])})
    result = detect_anomalies(df, contamination=0.01)
    assert result['anomaly'].iloc[-1] == 1, "The spike at index 299 must be flagged as anomaly"


def test_detect_anomalies_score_spike_is_highest():
    """The anomaly score for the spike should be the highest in the dataset."""
    np.random.seed(1)
    signal = np.concatenate([np.random.randn(299), [1000.0]])
    df = pd.DataFrame({'y': signal})
    result = detect_anomalies(df, contamination=0.01)
    assert result['anomaly_score'].idxmax() == 299, \
        "The spike at index 299 must have the highest anomaly score"


# ==============================
# forecast.py — run_forecast()
# ==============================

def test_forecast_returns_yhat():
    """run_forecast() must return a dataframe containing a 'yhat' column."""
    dates    = pd.date_range(start='2024-01-01', periods=200, freq='s')
    train_df = pd.DataFrame({'ds': dates, 'y': np.random.randn(200)})
    forecast = run_forecast(train_df, steps=20, print_changepoints=False)
    assert 'yhat' in forecast.columns, "Forecast must contain 'yhat'"


def test_forecast_includes_confidence_interval():
    """run_forecast() must return uncertainty bounds yhat_lower and yhat_upper."""
    dates    = pd.date_range(start='2024-01-01', periods=200, freq='s')
    train_df = pd.DataFrame({'ds': dates, 'y': np.random.randn(200)})
    forecast = run_forecast(train_df, steps=20, print_changepoints=False)
    assert 'yhat_lower' in forecast.columns, "Must include lower confidence bound"
    assert 'yhat_upper' in forecast.columns, "Must include upper confidence bound"


def test_forecast_lower_below_upper():
    """yhat_lower must always be less than or equal to yhat_upper."""
    dates    = pd.date_range(start='2024-01-01', periods=200, freq='s')
    train_df = pd.DataFrame({'ds': dates, 'y': np.random.randn(200)})
    forecast = run_forecast(train_df, steps=20, print_changepoints=False)
    assert (forecast['yhat_lower'] <= forecast['yhat_upper']).all(), \
        "yhat_lower must be <= yhat_upper for all rows"


def test_forecast_correct_number_of_future_steps():
    """run_forecast() must return exactly train_len + steps rows in the forecast."""
    n_train  = 200
    n_steps  = 30
    dates    = pd.date_range(start='2024-01-01', periods=n_train, freq='s')
    train_df = pd.DataFrame({'ds': dates, 'y': np.random.randn(n_train)})
    forecast = run_forecast(train_df, steps=n_steps, print_changepoints=False)
    assert len(forecast) == n_train + n_steps, \
        f"Expected {n_train + n_steps} rows, got {len(forecast)}"


# ==============================
# lstm.py — build_sequences() + run_lstm()
# ==============================

def test_build_sequences_correct_shape():
    """build_sequences() must produce X with shape (n - window, window, 1)."""
    data = np.random.randn(100, 1)
    X, y = build_sequences(data, window_size=10)
    assert X.shape == (90, 10, 1), f"Expected (90, 10, 1), got {X.shape}"
    assert y.shape == (90, 1),    f"Expected (90, 1), got {y.shape}"


def test_lstm_output_same_shape():
    """run_lstm() must return y_test and y_pred with identical shapes."""
    series = pd.Series(np.random.randn(600))
    y_test, y_pred = run_lstm(series)
    assert y_test.shape == y_pred.shape, \
        f"y_test and y_pred must have the same shape: {y_test.shape} vs {y_pred.shape}"


def test_lstm_output_is_1d():
    """run_lstm() must return 1D arrays, not 2D."""
    series = pd.Series(np.random.randn(600))
    y_test, y_pred = run_lstm(series)
    assert y_test.ndim == 1, "y_test must be 1D"
    assert y_pred.ndim == 1, "y_pred must be 1D"


def test_lstm_raises_on_short_series():
    """run_lstm() must raise ValueError when the series is too short for the window."""
    short_series = pd.Series(np.random.randn(10))
    with pytest.raises(ValueError, match="too short"):
        run_lstm(short_series)
