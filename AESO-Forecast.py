"""
aeso_forecasting
-----------------------------------
Forecasts hourly AESO pool prices using MSTL + AutoARIMA + SeasonalNaive.
Optimized for readability, efficiency, and modularity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, SeasonalNaive
from utilsforecast.plotting import plot_series

# =============================
# CONFIGURATION
# =============================

CSV_PATH = "//Users//alexzhang//Downloads//HistoricalPoolPriceReport.csv"
FORECAST_HOURS = 24  # Forecast horizon (hours)
SEASONALITIES = [24, 24 * 7]  # daily + weekly
CONFIDENCE_LEVEL = [90]  # prediction intervals
FREQ = "h"  # hourly data


# =============================
# DATA LOADING & PREPARATION
# =============================

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Loads and cleans AESO historical price data."""
    df = pd.read_csv(csv_path)
    df.insert(0, "unique_id", "Electricity_Hourly")

    # Keep only relevant columns
    df = df.rename(columns={"DateTime": "ds", "AIL Demand (MW)": "y"})
    drop_cols = [c for c in ["30Ravg ($)", "Price ($)"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Parse timestamps and drop invalid entries
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).reset_index(drop=True)

    # Ensure numeric target
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])

    print(f"✅ Loaded {len(df):,} valid hourly records.")
    return df


# =============================
# MODEL TRAINING & FORECASTING
# =============================

def train_and_forecast(df: pd.DataFrame, forecast_horizon: int = FORECAST_HOURS) -> pd.DataFrame:
    """Fits MSTL + SeasonalNaive models and produces forecasts."""
    mstl = MSTL(season_length=SEASONALITIES, trend_forecaster=AutoARIMA())
    models = [mstl, SeasonalNaive(season_length=24)]

    sf = StatsForecast(models=models, freq=FREQ)
    sf = sf.fit(df=df)

    forecasts = sf.predict(h=forecast_horizon, level=CONFIDENCE_LEVEL)
    print(f"✅ Forecast generated for next {forecast_horizon} hours.")
    return forecasts


# =============================
# VISUALIZATION
# =============================

def plot_forecasts(y_hist: pd.DataFrame, forecasts: pd.DataFrame, models: list[str]):
    """Plots actual and forecasted values with confidence intervals."""
    fig, ax = plt.subplots(figsize=(18, 6))
    y_hist = y_hist.tail(24 * 7)  # last week for context
    merged = y_hist.merge(forecasts, how="outer", on=["unique_id", "ds"])
    merged = merged.set_index("ds")

    # Plot each model forecast
    for model, color in zip(models, ["orange", "green", "red"]):
        ax.plot(merged.index, merged["y"], color="black", label="Actual")
        ax.plot(merged.index, merged[model], color=color, label=f"{model} Forecast", linewidth=2)
        if f"{model}-lo-90" in merged.columns:
            ax.fill_between(
                merged.index,
                merged[f"{model}-lo-90"],
                merged[f"{model}-hi-90"],
                color=color,
                alpha=0.3,
                label=f"{model} 90% CI",
            )

    ax.set_title("AESO Hourly Electricity Forecast", fontsize=18)
    ax.set_xlabel("Timestamp", fontsize=14)
    ax.set_ylabel("Electricity Demand (MW)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================
# MAIN EXECUTION
# =============================

if __name__ == "__main__":
    start_time = time()

    df = load_and_prepare_data(CSV_PATH)

    # Train-test split (last 24 hours as test)
    df_train, df_test = df[:-FORECAST_HOURS], df[-FORECAST_HOURS:]

    forecasts = train_and_forecast(df_train, forecast_horizon=FORECAST_HOURS)

    elapsed = (time() - start_time) / 60
    print(f" Training + forecasting completed in {elapsed:.2f} minutes.")

    plot_forecasts(df, forecasts, models=["MSTL", "SeasonalNaive"])
