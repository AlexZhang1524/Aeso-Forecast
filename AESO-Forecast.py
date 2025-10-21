import urllib.request
import ssl
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# ---------------------------------------------------------------------
# 1. FETCH AESO DATA USING urllib
# ---------------------------------------------------------------------
def fetch_aeso_data(days_back=7):
    import ssl
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    # ✅ Correct AESO public report endpoint
    url = f"https://api.aeso.ca/report/v1.1/price/poolPrice?startDate={start_date}&endDate={end_date}"
    headers = {
        "Cache-Control": "no-cache",
        "Accept": "application/json"
    }

    print(f"🔌 Fetching AESO pool prices from {start_date} to {end_date}...")

    req = urllib.request.Request(url, headers=headers)
    ssl_context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(req, context=ssl_context) as response:
            print(f"✅ Response code: {response.getcode()}")
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"❌ Error fetching AESO data: {e}")
        return pd.DataFrame()

    pool_data = data.get("return", {}).get("poolPrice", [])
    if not pool_data:
        print("⚠️ No pool price data returned.")
        print("🔍 Response preview:")
        print(json.dumps(data, indent=2)[:1000])
        return pd.DataFrame()

    df = pd.DataFrame(pool_data)
    df.rename(columns={"begin_datetime_mpt": "ds", "pool_price": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().sort_values("ds").reset_index(drop=True)

    print(f"✅ Retrieved {len(df)} records from AESO.")
    return df


# ---------------------------------------------------------------------
# 2. SIMPLE FORECAST MODEL
# ---------------------------------------------------------------------
def forecast_prices(df, hours_ahead=24):
    """
    Simple forecast using linear trend + noise.
    """
    if df.empty:
        print("⚠️ No data to forecast.")
        return pd.DataFrame()

    print("📈 Generating simple forecast...")

    df_recent = df.tail(48)
    x = np.arange(len(df_recent))
    y = df_recent["y"].values
    coef = np.polyfit(x, y, 1)
    trend = np.poly1d(coef)

    forecast_x = np.arange(len(df_recent), len(df_recent) + hours_ahead)
    forecast_y = trend(forecast_x) + np.random.normal(0, 2, hours_ahead)

    forecast_dates = [df["ds"].iloc[-1] + timedelta(hours=i+1) for i in range(hours_ahead)]
    df_forecast = pd.DataFrame({"ds": forecast_dates, "forecast": forecast_y})

    print("✅ Forecast generated successfully.")
    return df_forecast


# ---------------------------------------------------------------------
# 3. PLOT RESULTS
# ---------------------------------------------------------------------
def plot_results(df, df_forecast):
    plt.figure(figsize=(14, 6))
    plt.plot(df["ds"], df["y"], label="Actual Pool Price", linewidth=2)
    if not df_forecast.empty:
        plt.plot(df_forecast["ds"], df_forecast["forecast"], linestyle="--", label="Forecast (Next 24h)")
    plt.title("AESO Pool Price (Historical + Forecast)", fontsize=16)
    plt.xlabel("Date/Time (MPT)")
    plt.ylabel("Price ($/MWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------------------
def main():
    print("🚀 Starting AESO Forecasting Pipeline...")
    df_live = fetch_aeso_data(days_back=7)

    if df_live.empty:
        print("⚠️ No data to process.")
        return

    df_live.to_csv("AESO_PoolPrice_History.csv", index=False)
    print("📁 Saved AESO_PoolPrice_History.csv")

    df_forecast = forecast_prices(df_live)
    df_forecast.to_csv("AESO_PoolPrice_Forecast.csv", index=False)
    print("📁 Saved AESO_PoolPrice_Forecast.csv")

    plot_results(df_live, df_forecast)


if __name__ == "__main__":
    main()
