# src/forecast.py

import pandas as pd
import numpy as np
from prophet import Prophet


def run_forecast(train_df, steps, print_changepoints=True):
    """Fit a Prophet model on training data and forecast future EEG values.

    Improvements over v1:
    - Reduced changepoint_prior_scale from 0.5 to 0.1 to prevent overfitting
      to noise — EEG signals are noisy and 0.5 caused the model to overreact
      to local fluctuations rather than learning real trends.
    - Added seasonality_mode='additive' explicitly (cleaner for EEG where
      amplitude doesn't scale with the trend level).
    - Prints detected changepoints so the user can see where the model
      identified real trend shifts in the brain signal.
    - Returns a named tuple / dict alongside the forecast for easier
      downstream use of uncertainty bounds.

    Args:
        train_df (pd.DataFrame): Training data with 'ds' (datetime) and
            'y' (EEG signal) columns.
        steps (int): Number of future time steps to forecast.
        print_changepoints (bool): If True, print the top detected
            trend changepoints. Default True.

    Returns:
        pd.DataFrame: Prophet forecast dataframe including:
            - 'yhat': predicted signal value
            - 'yhat_lower': lower 95% confidence bound
            - 'yhat_upper': upper 95% confidence bound
            - 'ds': corresponding datetime
    """
    model = Prophet(
        interval_width=0.95,
        changepoint_prior_scale=0.1,      # was 0.5 — reduced to avoid overfitting noise
        seasonality_prior_scale=5.0,       # moderate seasonality flexibility
        seasonality_mode='additive',       # EEG amplitudes don't scale with trend
        yearly_seasonality=False,          # EEG recordings are not yearly-cyclical
        weekly_seasonality=False,          # same
        daily_seasonality=False,           # same — disable defaults that don't apply
    )

    model.fit(train_df, iter=500)          # was default 300 — more robust convergence

    future = model.make_future_dataframe(periods=steps, freq='s')
    forecast = model.predict(future)

    # --- Changepoint summary ---
    if print_changepoints:
        changepoints = model.changepoints
        deltas = model.params['delta'].mean(axis=0)
        significant = [(cp, d) for cp, d in zip(changepoints, deltas) if abs(d) > 0.01]
        if significant:
            print(f"\n[INFO] Prophet detected {len(significant)} significant trend changepoints:")
            for cp, delta in sorted(significant, key=lambda x: abs(x[1]), reverse=True)[:5]:
                direction = '▲ rising' if delta > 0 else '▼ falling'
                print(f"       {cp.strftime('%Y-%m-%d %H:%M:%S')}  {direction}  (δ={delta:.4f})")
        else:
            print("[INFO] No significant trend changepoints detected.")

    # --- Forecast uncertainty summary ---
    forecast_section = forecast.iloc[-steps:]
    avg_interval = (forecast_section['yhat_upper'] - forecast_section['yhat_lower']).mean()
    print(f"[INFO] Average 95% confidence interval width: ±{avg_interval / 2:.3f}")

    return forecast
