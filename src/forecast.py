# src/forecast.py

from prophet import Prophet

def run_forecast(df):
    model = Prophet(
        interval_width=0.95,
        changepoint_prior_scale=0.5
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=50, freq='s')
    forecast = model.predict(future)

    return forecast
