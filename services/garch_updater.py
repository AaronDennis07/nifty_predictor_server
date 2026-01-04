# ==========================================================
# DAILY GARCH UPDATER (CONDITION-BASED)
# ==========================================================

import pandas as pd
import numpy as np
import joblib
from arch import arch_model
from datetime import datetime

# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "../data/combined_sorted_data.csv"
STATE_PATH = "../models/latest_garch_vol.pkl"

ROLLING_WINDOW = 500        # ~2 years of trading days
VOL_DEVIATION_THRESHOLD = 0.30
FORCE_UPDATE_WEEKDAY = 4    # Friday (0=Mon ... 4=Fri)

# ==========================================================
# HELPERS
# ==========================================================

def load_returns():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"],
        dayfirst=True
    ).set_index("Date").sort_index()

    df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()


def realized_vol_5d(log_returns):
    return log_returns.tail(5).std() * np.sqrt(252)


def should_update_garch(today, realized_vol, garch_vol):
    # Weekly scheduled update
    if today.weekday() == FORCE_UPDATE_WEEKDAY:
        return True, "WEEKLY_REFRESH"

    # Regime break trigger
    if abs(realized_vol - garch_vol) / garch_vol > VOL_DEVIATION_THRESHOLD:
        return True, "REGIME_BREAK"

    return False, "NO_UPDATE"


def train_garch(log_returns):
    model = arch_model(
        log_returns * 100,
        mean="Zero",
        vol="Garch",
        p=1,
        q=1
    )

    res = model.fit(disp="off")
    forecast = res.forecast(horizon=5)

    vol = np.sqrt(forecast.variance.iloc[-1].mean())
    vol = vol * np.sqrt(252) / 100

    return float(vol)

# ==========================================================
# MAIN ENTRY (CALLED DAILY)
# ==========================================================

def update_garch_daily():

    today = datetime.now()

    # Load returns
    df = load_returns()
    log_returns = df["Log_Ret"].iloc[-ROLLING_WINDOW:]

    # Load last GARCH state
    try:
        garch_vol = joblib.load(STATE_PATH)
    except FileNotFoundError:
        print("No existing GARCH state found. Training fresh GARCH...")
        garch_vol = train_garch(log_returns)
        joblib.dump(garch_vol, STATE_PATH)
        return {
            "status": "INITIALIZED",
            "garch_vol": round(garch_vol, 4)
        }

    # Compute realized volatility
    rv_5d = realized_vol_5d(log_returns)

    # Decide whether to retrain
    retrain, reason = should_update_garch(today, rv_5d, garch_vol)

    if retrain:
        new_garch_vol = train_garch(log_returns)
        joblib.dump(new_garch_vol, STATE_PATH)

        return {
            "status": "UPDATED",
            "reason": reason,
            "old_garch_vol": round(garch_vol, 4),
            "new_garch_vol": round(new_garch_vol, 4),
            "realized_vol_5d": round(rv_5d, 4)
        }

    return {
        "status": "SKIPPED",
        "reason": reason,
        "garch_vol": round(garch_vol, 4),
        "realized_vol_5d": round(rv_5d, 4)
    }


# ==========================================================
# CLI / CRON ENTRY
# ==========================================================

if __name__ == "__main__":
    result = update_garch_daily()
    print(result)
