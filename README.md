# 🧠 Smart EEG Forecast AI

## Overview

This project is an AI-based system that forecasts future EEG (brain signal) activity and detects anomalies in real-time. It helps identify unusual brain patterns early and provides insights into signal behavior. The system is designed for researchers and non-experts who need simple and reliable forecasting tools.

---

## Problem Statement

Understanding future trends in brain activity is challenging due to noisy and complex EEG signals. Traditional approaches rely only on past data and fail to detect early warning signs.

---

## Features

* 📈 Forecast future EEG signal values
* 📊 Confidence interval (uncertainty range)
* ⚖️ Baseline comparison using moving average
* 🚨 Anomaly detection (spikes/drops)
* 🧠 Simple explanation for non-experts

---

## Tech Stack

* Python
* Pandas, NumPy
* Prophet (forecasting)
* Scikit-learn (anomaly detection)
* Matplotlib (visualization)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python src/main.py
```

---

## Usage

* Input: EEG dataset (CSV format)
* Output:

  * Forecast graph
  * Detected anomalies
  * Model insights
  * Performance metrics

---

## Example Output

* EEG trend shows gradual increase
* 21 anomalies detected
* MAE: 11.57
* RMSE: 14.44

---

## Architecture

Data → Preprocessing → Forecasting (Prophet) → Anomaly Detection → Evaluation → Visualization

---

## Evaluation

The model is evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

A moving average baseline is used for comparison to ensure reliability and avoid overfitting.

---

## Key Insight

The baseline model performs competitively due to the high noise in EEG signals. This highlights the importance of simple models as strong benchmarks and ensures transparency in predictions.

---

## Limitations

* Works on single-channel EEG data
* Performance affected by noisy signals

---

## Future Improvements

* Multi-channel EEG analysis
* Deep learning models (LSTM)
* Real-time dashboard

---

## Conclusion

This system provides a lightweight and interpretable approach to forecasting EEG signals and detecting anomalies, enabling early detection of unusual brain activity patterns.

## 📊 Model Comparison

### 🔹 Prophet vs Baseline

![Prophet vs Baseline](assets/output_comparison.png)

This graph compares the forecasting performance of the Prophet model with a simple moving average baseline.

* Prophet captures overall trends and patterns in EEG signals
* Baseline provides a simple smoothing reference
* Results show that while Prophet models trends, baseline performs competitively due to high noise in EEG data

---

### 🔹 LSTM Model Comparison

![LSTM Comparison](assets/output_lstm.png)

This graph shows predictions from the LSTM deep learning model compared to actual EEG signals.

* LSTM captures temporal dependencies using sequence learning
* However, performance is sensitive to noise and requires more training data
* Compared to Prophet, LSTM is less stable for this dataset

---

## 🧠 Key Insight

Although advanced models like LSTM were explored, Prophet was chosen as the primary model due to:

* Better interpretability
* Faster training
* Robust performance on noisy EEG data

This demonstrates the importance of selecting models based on data characteristics rather than complexity.

