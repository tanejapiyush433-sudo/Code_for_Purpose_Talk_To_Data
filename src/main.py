# src/main.py

import os
import sys
import zipfile
import glob

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Save plots to files instead of blocking with plt.show()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error

from lstm import run_lstm
from forecast import run_forecast
from anomaly import detect_anomalies
from utils import preprocess, compute_band_powers, BANDS


# ==============================
# CONFIGURATION
# ==============================
SAMPLE_RATE_HZ = 256          # Change to match your EEG device (e.g. 128, 256, 512)
CHANNEL = None                # Set to a column name e.g. 'Fp1' to pick a specific channel
BASELINE_WINDOW = 10          # Rolling window size for moving average baseline
TRAIN_RATIO = 0.8             # Fraction of data used for training
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')


# ==============================
# LOAD DATA
# ==============================
def load_data():
    """Load EEG CSV data from a zip archive or directly from a CSV file.

    Searches for data in the following order:
    1. ZIP file containing CSVs (default: ./data/)
    2. CSV files directly in ./data/

    Returns:
        pd.DataFrame: Raw loaded dataframe.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    # Try zip files first
    zip_files = glob.glob(os.path.join(data_dir, '*.zip'))
    if zip_files:
        with zipfile.ZipFile(zip_files[0], 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if csv_files:
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
                    print(f"[INFO] Loaded from ZIP: {zip_files[0]} → {csv_files[0]}")
                    return df

    # Fallback: direct CSV files
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        print(f"[INFO] Loaded CSV: {csv_files[0]}")
        return df

    raise FileNotFoundError(
        "No CSV or ZIP data file found in the data/ directory.\n"
        "Place your EEG CSV file in the data/ folder and re-run."
    )


def ensure_output_dir():
    """Create the outputs/ directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Plots will be saved to: {OUTPUT_DIR}/")


# ==============================
# MAIN PIPELINE
# ==============================
def main():
    ensure_output_dir()

    # --- Load ---
    raw_df = load_data()
    print(f"[INFO] Raw data shape: {raw_df.shape}")

    # --- Preprocess (filter + channel select + band powers) ---
    df = preprocess(raw_df, channel=CHANNEL, fs=SAMPLE_RATE_HZ)

    # --- Baseline ---
    df['baseline'] = df['y'].rolling(window=BASELINE_WINDOW).mean()

    # --- Train/test split ---
    split_index = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_index]
    test_df  = df.iloc[split_index:]
    print(f"\n[INFO] Train samples: {len(train_df)} | Test samples: {len(test_df)}")

    # --- Forecast ---
    print("\n[INFO] Fitting Prophet model...")
    forecast = run_forecast(train_df, steps=len(test_df))

    # --- Align predictions to test set ---
    forecast_test  = forecast.iloc[-len(test_df):]
    y_true         = test_df['y'].reset_index(drop=True)
    y_pred_model   = forecast_test['yhat'].reset_index(drop=True)
    y_pred_lower   = forecast_test['yhat_lower'].reset_index(drop=True)
    y_pred_upper   = forecast_test['yhat_upper'].reset_index(drop=True)
    y_pred_baseline = test_df['baseline'].reset_index(drop=True)

    # --- Anomaly detection ---
    print("\n[INFO] Running anomaly detection...")
    df = detect_anomalies(df)

    # --- Evaluation ---
    mae_model  = mean_absolute_error(y_true, y_pred_model)
    rmse_model = np.sqrt(mean_squared_error(y_true, y_pred_model))

    baseline_filled = y_pred_baseline.fillna(y_pred_baseline.mean())
    mae_baseline  = mean_absolute_error(y_true, baseline_filled)
    rmse_baseline = np.sqrt(mean_squared_error(y_true, baseline_filled))

    anomaly_count = df['anomaly'].sum()
    stress_pct = (df['anomaly'].sum() / len(df)) * 100

    print("\n" + "=" * 46)
    print("           TEST EVALUATION RESULTS")
    print("=" * 46)
    print(f"\n{'Model':<28} {'MAE':>7}  {'RMSE':>7}")
    print("-" * 46)
    print(f"{'Prophet (primary)':<28} {mae_model:>7.4f}  {rmse_model:>7.4f}")
    print(f"{'Baseline (moving average)':<28} {mae_baseline:>7.4f}  {rmse_baseline:>7.4f}")
    print("-" * 46)
    improvement = (mae_baseline - mae_model) / mae_baseline * 100
    print(f"\nProphet vs Baseline improvement: {improvement:+.1f}% MAE")
    print(f"Anomalies detected: {int(anomaly_count)} ({stress_pct:.2f}% of signal)")
    print("=" * 46)

    # --- Frequency band bar chart ---
    band_powers = compute_band_powers(df['y'].values, fs=SAMPLE_RATE_HZ)
    _plot_band_powers(band_powers)

    # --- LSTM comparison ---
    print("\n[INFO] Training LSTM model...")
    y_test_lstm, y_pred_lstm = run_lstm(df['y'])
    _plot_lstm(y_test_lstm, y_pred_lstm)

    # --- Prophet vs Baseline with anomalies ---
    _plot_forecast(y_true, y_pred_model, y_pred_lower, y_pred_upper,
                   y_pred_baseline, df)

    print("\n[INFO] Pipeline complete. All plots saved to outputs/")


# ==============================
# PLOT HELPERS
# ==============================

def _plot_band_powers(band_powers):
    """Bar chart of EEG frequency band relative power."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bands   = list(band_powers.keys())
    powers  = list(band_powers.values())
    colors  = ['#5B8DB8', '#6BAF92', '#F0A500', '#E05C5C', '#9B59B6']
    bars    = ax.bar(bands, powers, color=colors, edgecolor='white', linewidth=0.8)

    ax.set_title('EEG Frequency Band Power Distribution', fontsize=13, pad=10)
    ax.set_ylabel('Relative Power (fraction of total)')
    ax.set_ylim(0, max(powers) * 1.3)
    ax.spines[['top', 'right']].set_visible(False)

    for bar, pwr in zip(bars, powers):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{pwr:.3f}', ha='center', va='bottom', fontsize=10)

    dominant = max(band_powers, key=band_powers.get)
    state = {'delta': 'deep sleep', 'theta': 'drowsy', 'alpha': 'relaxed',
             'beta': 'alert/stressed', 'gamma': 'high cognitive load'}.get(dominant, dominant)
    ax.set_xlabel(f'Dominant band: {dominant.upper()} ({state})', fontsize=10, color='#555')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'band_powers.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def _plot_lstm(y_test, y_pred):
    """Line plot comparing LSTM predictions to actual EEG signal."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y_test, label='Actual signal', color='#2C3E50', linewidth=1)
    ax.plot(y_pred, label='LSTM prediction', color='#E74C3C', linewidth=1, alpha=0.85)
    ax.set_title('LSTM Model — Predicted vs Actual EEG Signal (Test Set)', fontsize=13)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Signal amplitude')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'lstm_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def _plot_forecast(y_true, y_pred, y_lower, y_upper, y_baseline, full_df):
    """Multi-panel plot: forecast + confidence interval + anomaly markers."""
    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

    # --- Top panel: forecast ---
    ax1 = fig.add_subplot(gs[0])
    x   = range(len(y_true))

    ax1.fill_between(x, y_lower, y_upper, alpha=0.15, color='#3498DB', label='95% confidence interval')
    ax1.plot(x, y_true.values,        color='#2C3E50', linewidth=1,    label='Actual EEG signal')
    ax1.plot(x, y_pred.values,        color='#3498DB', linewidth=1.2,  label='Prophet forecast')
    ax1.plot(x, y_baseline.values,    color='#95A5A6', linewidth=1,
             linestyle='--', label='Moving average baseline')

    # Mark anomalies that fall in the test window
    test_anomalies = full_df.iloc[-len(y_true):]
    anom_idx = test_anomalies.reset_index(drop=True)
    anom_mask = anom_idx['anomaly'] == 1
    if anom_mask.any():
        ax1.scatter(anom_mask[anom_mask].index, y_true[anom_mask],
                    color='#E74C3C', zorder=5, s=20, label='Anomaly', alpha=0.8)

    ax1.set_title('Test Set: Prophet Forecast vs Actual EEG Signal', fontsize=13, pad=10)
    ax1.set_ylabel('Signal amplitude')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.spines[['top', 'right']].set_visible(False)

    # --- Bottom panel: residuals ---
    ax2 = fig.add_subplot(gs[1])
    residuals = y_true.values - y_pred.values
    ax2.bar(x, residuals, color=np.where(residuals >= 0, '#3498DB', '#E74C3C'),
            alpha=0.5, width=1.0)
    ax2.axhline(0, color='#2C3E50', linewidth=0.8)
    ax2.set_title('Forecast Residuals (Actual − Predicted)', fontsize=11)
    ax2.set_xlabel('Time step (test set)')
    ax2.set_ylabel('Residual')
    ax2.spines[['top', 'right']].set_visible(False)

    path = os.path.join(OUTPUT_DIR, 'prophet_forecast.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


# ==============================
# ENTRY POINT
# ==============================
if __name__ == '__main__':
    main()
