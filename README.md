# 🧠 Smart EEG Forecast AI — NatWest Code for Purpose

## Purpose & Financial Wellbeing

This project was built for NatWest's **Code for Purpose** challenge, tackling the problem of financial vulnerability through neuroscience.

Research shows that financial stress directly alters brainwave activity — elevated **beta waves** (13–30 Hz) are a measurable marker of anxiety and cognitive overload. People in high-stress cognitive states are more likely to make poor financial decisions, miss repayments, or fall victim to scams.

Our system analyses EEG (electroencephalogram) brain signals to:

- **Detect stress episodes** — anomaly spikes in the EEG signal can indicate moments of cognitive overload or emotional distress, flagging when a customer may be in a vulnerable state before making significant financial decisions
- **Forecast future signal trends** — anticipate when a person's stress level is building, enabling proactive intervention (e.g. a wellbeing prompt, simplified banking UX, or a call from a financial health advisor)
- **Quantify cognitive state** — frequency band power analysis reveals whether a user is relaxed (alpha dominant), alert (beta dominant), or fatigued (theta dominant) — enabling truly personalised banking experiences
- **Run without clinical infrastructure** — the lightweight pipeline is compatible with consumer-grade EEG headsets (Muse, OpenBCI, Emotiv), not hospital equipment

This directly supports NatWest's commitment to financial inclusion and customer wellbeing — going beyond transactions to understand the human behind the account.

---

## Overview

An AI pipeline that forecasts future EEG brain signal activity, detects cognitive anomalies, and characterises mental state through frequency band analysis. Designed to be interpretable by non-experts while maintaining rigorous signal processing standards.

---

## What Changed in Round 2

| Feature | Round 1 | Round 2 |
|---|---|---|
| Signal preprocessing | Raw signal, no filtering | Butterworth bandpass filter (0.5–40 Hz) |
| Channel support | First column only | Multi-channel, user-selectable |
| Anomaly detection | Hard-coded contamination (1%) | Dynamic contamination from signal kurtosis |
| Anomaly output | Binary flag only | Flag + severity score + top anomaly report |
| LSTM architecture | 1 layer, 5 epochs, window=10 | 2 layers + Dropout, early stopping, window=30 |
| Frequency analysis | None | Delta/Theta/Alpha/Beta/Gamma band powers |
| Forecast tuning | changepoint_prior=0.5 (overfitting) | Reduced to 0.1, additive seasonality |
| Visualisation | 2 basic plots | 3 plots incl. residuals, band power bar chart, anomaly markers |
| Test coverage | 6 tests (happy path only) | 18 tests incl. edge cases, new functions, shapes |
| NaN handling | Not handled | Forward-fill with warning |
| Output | Terminal only | PNG files saved to outputs/ |

---

## Features

- 🔬 **Bandpass filtering** (0.5–40 Hz Butterworth) removes power-line noise and muscle artefacts before modelling — a standard step in clinical EEG pipelines
- 📊 **Frequency band power analysis** — quantifies delta, theta, alpha, beta, and gamma relative power using Welch's method, with a dominant band summary
- 📈 **EEG forecasting** using Prophet with tuned parameters, printed changepoints, and 95% confidence intervals
- ⚖️ **Baseline comparison** — rolling mean benchmark to validate that the model adds value beyond a simple heuristic
- 🚨 **Intelligent anomaly detection** — Isolation Forest with kurtosis-derived contamination rate and per-anomaly severity scoring
- 🧠 **Stacked LSTM** — two-layer LSTM with Dropout and early stopping for better temporal modelling
- 📉 **Residual analysis** — bottom panel on the forecast plot shows where the model over- or under-predicts
- 💾 **Saved PNG outputs** — all plots saved to `outputs/` instead of blocking with `plt.show()`

---

## Tech Stack

- Python 3.10+
- Pandas, NumPy, SciPy
- Prophet (primary forecasting model)
- Scikit-learn (Isolation Forest anomaly detection)
- TensorFlow / Keras (stacked LSTM)
- Matplotlib (visualisation)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the Project

Place your EEG data file (CSV or ZIP containing CSV) in the `data/` folder, then run from the project root:

```bash
python src/main.py
```

To use a specific EEG channel, edit `CHANNEL` at the top of `src/main.py`:

```python
CHANNEL = 'Fp1'           # Use the Fp1 electrode channel
SAMPLE_RATE_HZ = 256      # Set to match your EEG device
```

---

## Usage

**Input:** EEG dataset placed in the `data/` folder (CSV or ZIP containing CSV).
Any multi-column CSV is supported — the pipeline will list all available channels and use the first by default.

**Output (saved to `outputs/`):**

| File | Description |
|---|---|
| `band_powers.png` | Bar chart of delta/theta/alpha/beta/gamma relative power |
| `prophet_forecast.png` | Forecast vs actual with confidence interval, anomaly markers, and residuals panel |
| `lstm_comparison.png` | LSTM predicted vs actual EEG signal on the test set |

**Terminal output includes:**

- Channel selection and available channels
- Bandpass filter confirmation
- EEG band power summary with ASCII bar chart
- Prophet changepoint report
- Anomaly count, severity, and timestamps of top anomalies
- Evaluation table: Prophet vs Baseline MAE/RMSE
- LSTM MAE and RMSE

---

## Example Output

```
[INFO] Using EEG channel : 'Fp1'
[INFO] Other channels available: ['Fp2', 'F3', 'F4', 'C3', 'C4']
[INFO] Applying bandpass filter: 0.5–40 Hz @ 256 Hz sample rate

[INFO] EEG Band Power Summary:
       delta  0.312  ████████████
       theta  0.185  ███████
       alpha  0.241  █████████
       beta   0.198  ████████
       gamma  0.064  ██
[INFO] Dominant band: ALPHA (relaxed)

[INFO] Signal kurtosis: 3.412 → contamination set to 0.012
[INFO] Anomalies detected: 24 / 2048 samples (1.17%)
[INFO] Top 3 most severe anomalies:
       2024-01-01 00:14:22  signal=87.431  score=0.2841
       2024-01-01 00:07:09  signal=81.006  score=0.2613
       2024-01-01 00:21:55  signal=-74.22  score=0.2480

==========================================
           TEST EVALUATION RESULTS
==========================================

Model                          MAE     RMSE
------------------------------------------
Prophet (primary)           11.2341  13.9812
Baseline (moving average)   12.3001  15.1043
------------------------------------------

Prophet vs Baseline improvement: +8.7% MAE
Anomalies detected: 24 (1.17% of signal)
==========================================

[INFO] LSTM trained for 23 epochs (early stopping patience=7)
[INFO] LSTM Test MAE:  10.8821
[INFO] LSTM Test RMSE: 13.4412
```

---

## Architecture

```
EEG CSV / ZIP
     │
     ▼
Load Data (load_data)
     │
     ▼
Preprocess (utils.py)
  ├── Channel selection (multi-channel support)
  ├── NaN forward-fill
  ├── Bandpass filter 0.5–40 Hz (Butterworth)
  └── Frequency band power analysis (Welch)
     │
     ├─────────────────────────────────────┐
     ▼                                     ▼
Anomaly Detection (anomaly.py)     Train/Test Split (80/20)
  ├── Kurtosis-derived contamination       │
  ├── Isolation Forest (200 trees)    ┌────┴────┐
  └── Anomaly score ranking           ▼         ▼
                                  Forecast   LSTM
                               (forecast.py) (lstm.py)
                                  Prophet    Stacked 2-layer
                                  Tuned      + Dropout
                                  params     + Early stopping
                                      │         │
                                      └────┬────┘
                                           ▼
                                    Evaluation & Plots
                                  (MAE, RMSE, residuals,
                                   band chart, anomalies)
```

---

## Technical Depth

**Why Prophet over LSTM as the primary model?**
Prophet is interpretable, fast, and stable on noisy time series. For the NatWest use case, interpretability matters — a financial wellbeing system needs to explain *why* it raised a flag, not just that it did. Prophet's changepoint detection and decomposition (trend + seasonality) allow us to say "this anomaly coincides with a detected trend shift at 14:22" rather than producing a black-box score.

**Why Isolation Forest for anomaly detection?**
It is unsupervised (no labelled anomaly data needed), computationally efficient, and handles the non-Gaussian distribution of real EEG signals better than z-score or IQR methods. The contamination rate is now derived from the signal's kurtosis — spiky distributions (high kurtosis) produce a higher contamination rate automatically.

**Why a bandpass filter?**
Raw EEG is contaminated by 50/60 Hz power-line interference and muscle artefacts above 40 Hz. Without filtering, models partially learn to predict noise rather than brain activity. The Butterworth bandpass (0.5–40 Hz) is the standard preprocessing step in clinical EEG research (IEE, Emotiv, OpenBCI pipelines all apply this).

**Why frequency band analysis?**
The raw amplitude of an EEG signal is not clinically meaningful on its own. The *distribution of power across frequency bands* is what matters: elevated beta (13–30 Hz) indicates stress or anxiety, which is directly relevant to financial vulnerability detection.

---

## Evaluation

Models evaluated on the held-out 20% test set using:

- **MAE** — Mean Absolute Error (lower is better)
- **RMSE** — Root Mean Squared Error (penalises large errors more)

A moving average baseline ensures the models add genuine value. LSTM is compared head-to-head with Prophet on the same test window.

---

## Limitations

- Requires minimum ~512 samples for LSTM (window size = 30, needs 3× for sequences)
- Single-file EEG input; real deployments would stream data in real time
- Frequency band analysis assumes a 256 Hz sample rate by default — update `SAMPLE_RATE_HZ` in `main.py` to match your device

---

## Future Improvements

- Real-time streaming pipeline using WebSockets or Kafka
- Multi-session trend tracking — monitor a user's stress baseline over days/weeks
- Scenario forecasting: "what if alpha power drops below 0.15?"
- REST API wrapper so the pipeline can be embedded in a NatWest mobile app
- Personalised anomaly thresholds calibrated per individual rather than per dataset

---

## Conclusion

This system provides an interpretable, clinically-grounded approach to understanding brain activity patterns — with a direct line to NatWest's mission of supporting financial wellbeing. By combining robust signal processing, probabilistic forecasting, and anomaly detection, it can flag cognitive vulnerability before it becomes financial vulnerability.
