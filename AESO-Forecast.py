"""
aeso_live_forecast
-------------------------------------
Fetches live pool price / forecast data from AESO, fits two forecasting models
(Seasonal Naive vs MSTL+AutoARIMA), compares their accuracy, and outputs results.

"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, SeasonalNaive

# =============================
# CONFIGURATION
# =============================

API_BASE = "https://apimgw.aeso.ca/public"  # base AESO public API endpoint
API_KEY = os.getenv("AESO_API_KEY", "AESOForecast")     # set your AESO API key in env var
POOLPRICE_PATH = "/poolprice-api/v1.1/price/poolPrice"   # example endpoint
FREQ = "h"
SEASONALITIES = [24, 24 * 7]
FORECAST_HOURS = 24
CONF_LEVEL = [90]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# =============================
# LIVE DATA FETCHING
# =============================

def fetch_live_aeso_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch live AESO pool price data between start_date and end_date (YYYY-MM-DD).
    Returns: DataFrame with columns ['ds', 'y'] where ds=hour datetime, y=pool price.
    """
    if not API_KEY:
        raise RuntimeError("AESO_API_KEY not set in environment.")

    url = f"{API_BASE}{POOLPRICE_PATH}"
    params = {
        "startDate": start_date,
        "endDate": end_date
    }
    headers = {
        "X-API-KEY": API_KEY
    }
    logging.info(f"Requesting AESO pool price from {start_date} to {end_date}")
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    # The JSON structure may vary — you’ll need to inspect the "Return" or similar key
    # Example extracted fields (adjust as needed):
    records = j.get("return", {}).get("Pool Price Report", [])
    df = pd.DataFrame.from_records(records)
    df = df.rename(columns={
        "begin_datetime_mpt": "ds",
        "pool_price": "y"
    })
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    return df

# =============================
# MODELING
# =============================

def fit_and_forecast(df: pd.DataFrame, h: int = FORECAST_HOURS):
    """
    Fits two models on given time‐series df and forecasts h hours ahead.
    Returns: forecasts_df, metrics_df
    """
    # Seasonal Naive model
    snaive_model = SeasonalNaive(season_length=24)
    # MSTL + AutoARIMA
    mstl_model = MSTL(season_length=SEASONALITIES, trend_forecaster=AutoARIMA())

    sf = StatsForecast(models=[snaive_model, mstl_model], freq=FREQ)
    sf = sf.fit(df=df)

    forecasts = sf.predict(h=h, level=CONF_LEVEL)
    # rename model keys for clarity
    forecasts = forecasts.rename(columns={
        "SeasonalNaive": "SeasonalNaive",
        "MSTL": "MSTL"
    })

    # If you have true values for those horizon hours, calculate accuracy
    # Example: assume df_true exists with ds and y_true columns
    # Merge df_true and forecasts on ds, compute MAE/RMSE for each model
    # For now placeholders:
    metrics = {
        "SeasonalNaive_MAE": np.nan,
        "MSTL_MAE": np.nan,
        "SeasonalNaive_RMSE": np.nan,
        "MSTL_RMSE": np.nan
    }

    return forecasts, pd.DataFrame([metrics])

# =============================
# PLOTTING
# =============================

def plot_model_comparison(df_hist: pd.DataFrame,
                          forecasts: pd.DataFrame,
                          models: list[str]):
    """
    Plot historical data + forecasts for comparison of models.
    """
    df_plot = df_hist.tail(24 * 7).copy()
    df_plot = df_plot.set_index("ds")
    for model in models:
        df_plot[model] = np.nan

    # align forecast values
    for idx, row in forecasts.iterrows():
        ds = row["ds"]
        for model in models:
            df_plot.loc[ds, model] = row[model]

    plt.figure(figsize=(18, 6))
    plt.plot(df_plot.index, df_plot["y"], label="Actual", color="black", linewidth=2)
    colors = ["orange", "green"]
    for model, color in zip(models, colors):
        plt.plot(df_plot.index, df_plot[model], label=f"{model} Forecast", color=color, linewidth=2)
        if f"{model}-lo-90" in df_plot.columns:
            plt.fill_between(df_plot.index,
                             df_plot[f"{model}-lo-90"],
                             df_plot[f"{model}-hi-90"],
                             color=color,
                             alpha=0.3,
                             label=f"{model} 90% CI")

    plt.title("AESO Pool Price: Actual vs Forecasts", fontsize=18)
    plt.xlabel("Timestamp", fontsize=14)
    plt.ylabel("Pool Price ($/MWh)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================
# MAIN EXECUTION
# =============================

if __name__ == "__main__":
    # Step 1: Pull last‐few days live data
    today = datetime.utcnow().date()
    start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    df_live = fetch_live_aeso_prices(start, end)
    logging.info(f"Retrieved {len(df_live):,} records.")

    # Step 2: Fit & forecast
    forecasts, metrics = fit_and_forecast(df_live, h=FORECAST_HOURS)
    logging.info("Forecasts computed:")
    logging.info(forecasts.head())
    logging.info("Model metrics:")
    logging.info(metrics)

    # Step 3: Plot comparison
    plot_model_comparison(df_live, forecasts, models=["SeasonalNaive", "MSTL"])
