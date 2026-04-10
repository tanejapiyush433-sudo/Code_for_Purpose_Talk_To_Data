# src/forecast.py

from prophet import Prophet

def run_forecast(train_df, steps):
    model = Prophet(
        interval_width=0.95,
        changepoint_prior_scale=0.5
    )

    model.fit(train_df)

    future = model.make_future_dataframe(periods=steps, freq='s')
    forecast = model.predict(future)

    return forecast
