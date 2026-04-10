# src/forecast.py

from prophet import Prophet

def run_forecast(df):
    model = Prophet(
        interval_width=0.95,
        changepoint_prior_scale=0.5
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df), freq='s')
    forecast = model.predict(future)
    # ==============================
    # TEST PREDICTIONS
    # ==============================
    forecast_test = forecast.iloc[-len(test_df):]

    y_true = test_df['y'].reset_index(drop=True)
    y_pred_model = forecast_test['yhat'].reset_index(drop=True)
    y_pred_baseline = test_df['baseline'].reset_index(drop=True)

    return forecast
