# ==========================================================
# NIFTY AI : VOL + DIRECTION STACKED OPTIONS SYSTEM (INFERENCE)
# ==========================================================

import pandas as pd
import numpy as np
import joblib
from pandas.tseries.offsets import MonthEnd

# ==========================================================
# 1. LOAD MODELS & METADATA (ONCE)
# ==========================================================

vol_model = joblib.load("models/vol_model_xgb.pkl")
dir_model = joblib.load("models/dir_model_xgb.pkl")

vol_features = joblib.load("models/vol_features.pkl")
dir_features = joblib.load("models/dir_features.pkl")
latest_garch_vol = joblib.load("models/latest_garch_vol.pkl")

# ==========================================================
# 2. FEATURE ENGINEERING (BASE FEATURES)
# ==========================================================

def feature_engineering(df):
    df = df.copy()

    df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))

    df["HV_5"] = df["Log_Ret"].rolling(5).std() * np.sqrt(252)
    df["HV_20"] = df["Log_Ret"].rolling(20).std() * np.sqrt(252)

    df["Range_Vol"] = np.log(df["High"] / df["Low"])

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)

    df["ATR_Pct"] = tr.rolling(14).mean() / df["Close"]

    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Width"] = ((sma + 2 * std) - (sma - 2 * std)) / sma

    df["Ret_Sq"] = df["Log_Ret"] ** 2

    df["VIX_Change"] = df["India_VIX"].pct_change()
    df["VIX_Momentum"] = df["VIX_Change"].rolling(3).mean()
    df["VIX_vs_HV"] = df["India_VIX"] / (df["HV_20"] * 100)

    return df.dropna()

# ==========================================================
# 3. HELPERS
# ==========================================================

def last_thursday(date):
    d = pd.Timestamp(date) + MonthEnd(0)
    while d.weekday() != 3:
        d -= pd.Timedelta(days=1)
    return d


def round50(x):
    return int(round(x / 50) * 50)


def vol_regime(realized_vol_series, india_vix):
    slope = np.polyfit(range(len(realized_vol_series)), realized_vol_series, 1)[0]
    if india_vix > 14 or slope > 0:
        return "RISING_VOL"
    if slope < 0:
        return "FALLING_VOL"
    return "NEUTRAL_VOL"


def direction_regime(bull_prob, bear_prob):
    if bull_prob >= 0.65:
        return "BULLISH"
    if bear_prob >= 0.65:
        return "BEARISH"
    return "NEUTRAL"


def strategy(vol_reg, dir_reg):
    if vol_reg == "FALLING_VOL":
        if dir_reg == "BULLISH":
            return "BULL_PUT_SPREAD"
        if dir_reg == "BEARISH":
            return "BEAR_CALL_SPREAD"
        return "IRON_CONDOR"

    if vol_reg == "RISING_VOL":
        if dir_reg == "BULLISH":
            return "LONG_CALL"
        if dir_reg == "BEARISH":
            return "LONG_PUT"

    return "NO_TRADE"


def select_strikes(strategy, spot, bull_prob, bear_prob):
    atm = round50(spot)

    def dist(p):
        if p >= 0.75:
            return 150
        if p >= 0.65:
            return 100
        return 50

    if strategy == "LONG_CALL":
        return {"BUY_CALL": atm}

    if strategy == "LONG_PUT":
        return {"BUY_PUT": atm}

    if strategy == "BULL_PUT_SPREAD":
        d = dist(bull_prob)
        return {"SELL_PUT": atm - d, "BUY_PUT": atm - d - 300}

    if strategy == "BEAR_CALL_SPREAD":
        d = dist(bear_prob)
        return {"SELL_CALL": atm + d, "BUY_CALL": atm + d + 300}

    if strategy == "IRON_CONDOR":
        return {
            "SELL_PUT": atm - 200,
            "BUY_PUT": atm - 400,
            "SELL_CALL": atm + 200,
            "BUY_CALL": atm + 400
        }

    return None

# ==========================================================
# 4. EXPIRY LOGIC (IMPROVED)
# ==========================================================

def select_expiry_type(pred_vol, vol_regime, bull_prob, bear_prob):
    conviction = max(bull_prob, bear_prob)

    if vol_regime == "RISING_VOL" and conviction >= 0.65:
        return "WEEKLY"

    if vol_regime == "FALLING_VOL":
        return "MONTHLY"

    if pred_vol < 0.10 and conviction < 0.60:
        return "MONTHLY"

    return "NEXT_WEEKLY"


def select_expiry_date(trade_date, expiry_type):
    trade_date = pd.Timestamp(trade_date)

    if expiry_type == "WEEKLY":
        return trade_date + pd.offsets.Week(weekday=3)

    if expiry_type == "NEXT_WEEKLY":
        return trade_date + pd.offsets.Week(weekday=3) + pd.Timedelta(days=7)

    if expiry_type == "MONTHLY":
        return last_thursday(trade_date)

    return None

# ==========================================================
# 5. DO-NOT-TRADE FILTERS
# ==========================================================

def should_trade(
    signal_date,
    pred_vol,
    india_vix,
    bull_prob,
    bear_prob,
    vol_regime,
    dir_regime
):
    conviction = max(bull_prob, bear_prob)

    if conviction < 0.55:
        return False, "NO_EDGE"

    if pred_vol < 0.08 and india_vix < 10:
        return False, "DEAD_VOL"

    if vol_regime == "RISING_VOL" and dir_regime == "NEUTRAL":
        return False, "REGIME_CONFLICT"

    if pd.Timestamp(signal_date).weekday() == 3:
        return False, "EXPIRY_DAY"

    if pd.Timestamp(signal_date).weekday() == 0:
        return False, "MONDAY_GAP_RISK"

    return True, "TRADE_ALLOWED"

# ==========================================================
# 6. MAIN ENTRY — USED BY FASTAPI
# ==========================================================

def run_prediction():

    # -------------------------------
    # Load market data
    # -------------------------------
    nifty = pd.read_csv(
        "data/combined_sorted_data.csv",
        parse_dates=["Date"],
        dayfirst=True
    ).set_index("Date").sort_index()

    vix = pd.read_csv("data/combined_sorted_vix.csv")
    vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%m-%Y")
    vix = vix.set_index("Date").sort_index()
    vix["India_VIX"] = vix["Close"].astype(float).shift(1)

    data = nifty.join(vix[["India_VIX"]], how="left")
    data["India_VIX"] = data["India_VIX"].ffill()

    # -------------------------------
    # Feature engineering
    # -------------------------------
    data_feat = feature_engineering(data)
    latest = data_feat.iloc[-1:].copy()

    latest["Log_GARCH_Vol"] = np.log(latest_garch_vol)

    signal_date = latest.index[0]
    trade_date = signal_date + pd.Timedelta(days=1)

    # -------------------------------
    # MODEL A — Volatility
    # -------------------------------
    log_vol_pred = vol_model.predict(latest[vol_features])[0]
    pred_vol = float(np.exp(log_vol_pred))

    # -------------------------------
    # MODEL B — Direction
    # -------------------------------
    latest["Ret_5"] = data_feat["Log_Ret"].rolling(5).sum().iloc[-1]
    latest["Ret_20"] = data_feat["Log_Ret"].rolling(20).sum().iloc[-1]

    latest["Price_vs_SMA20"] = (
        latest["Close"].values[0] /
        data_feat["Close"].rolling(20).mean().iloc[-1] - 1
    )

    latest["Price_vs_SMA50"] = (
        latest["Close"].values[0] /
        data_feat["Close"].rolling(50).mean().iloc[-1] - 1
    )

    latest["Vol_Ratio"] = latest["HV_5"] / latest["HV_20"]
    latest["Range_Spike"] = latest["Range_Vol"] / latest["HV_20"]
    latest["Neg_Ret_Sq"] = np.minimum(latest["Log_Ret"], 0) ** 2
    latest["Log_Pred_Vol"] = np.log(pred_vol)

    missing = set(dir_features) - set(latest.columns)
    if missing:
        raise ValueError(f"Missing direction features: {missing}")

    probs = dir_model.predict_proba(latest[dir_features])[0]
    bear_prob = float(probs[0])
    bull_prob = float(probs[1])

    dreg = direction_regime(bull_prob, bear_prob)
    vreg = vol_regime(data_feat["HV_5"].iloc[-5:], latest["India_VIX"].values[0])

    # -------------------------------
    # DO NOT TRADE CHECK
    # -------------------------------
    trade_ok, reason = should_trade(
        signal_date,
        pred_vol,
        latest["India_VIX"].values[0],
        bull_prob,
        bear_prob,
        vreg,
        dreg
    )

    if not trade_ok:
        return {
            "signal_date": str(signal_date.date()),
            "trade_decision": "NO_TRADE",
            "reason": reason
        }

    # -------------------------------
    # Expiry & Strategy
    # -------------------------------
    expiry_type = select_expiry_type(pred_vol, vreg, bull_prob, bear_prob)
    expiry_date = select_expiry_date(trade_date, expiry_type)

    strat = strategy(vreg, dreg)
    legs = select_strikes(
        strat,
        latest["Close"].values[0],
        bull_prob,
        bear_prob
    )

    # -------------------------------
    # FINAL OUTPUT
    # -------------------------------
    return {
        "signal_date": str(signal_date.date()),
        "trade_date": str(trade_date.date()),
        "spot_close": float(round(latest["Close"].values[0], 2)),
        "predicted_volatility": round(pred_vol, 3),
        "india_vix": float(round(latest["India_VIX"].values[0], 2)),
        "bull_probability": round(bull_prob, 3),
        "bear_probability": round(bear_prob, 3),
        "vol_regime": vreg,
        "direction_regime": dreg,
        "strategy": strat,
        "expiry": {
            "type": expiry_type,
            "date": str(expiry_date.date())
        },
        "strikes": legs
    }


# Run locally
print(run_prediction())
