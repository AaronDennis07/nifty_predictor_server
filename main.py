from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import csv
import os
from datetime import datetime
from utils import read_csv_slice
from services.predictor import run_prediction

app = FastAPI(title="Market Data Feed API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# FILE PATHS
# ======================================================
NIFTY_CSV_FILE = "./data/combined_sorted_data.csv"
VIX_CSV_FILE = "./data/combined_sorted_vix.csv"


# ======================================================
# UTILS
# ======================================================
def ensure_newline_before_append(file_obj):
    """
    Ensures file ends with newline before appending.
    Fixes Windows + Excel CSV issues.
    """
    file_obj.seek(0, os.SEEK_END)
    if file_obj.tell() == 0:
        return
    file_obj.seek(file_obj.tell() - 1)
    if file_obj.read(1) != "\n":
        file_obj.write("\n")

def validate_date(date_str: str):
    try:
        datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Date must be in DD-MM-YYYY format"
        )

# ======================================================
# INIT CSV FILES
# ======================================================
def init_csv_files():
    if not os.path.exists(NIFTY_CSV_FILE):
        with open(NIFTY_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Shares Traded",
                "Turnover (₹ Cr)"
            ])

    if not os.path.exists(VIX_CSV_FILE):
        with open(VIX_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Prev. Close",
                "Change",
                "% Change"
            ])

init_csv_files()

# ======================================================
# MODELS
# ======================================================
class NiftyRow(BaseModel):
    Date: str
    Open: float
    High: float
    Low: float
    Close: float
    Shares_Traded: int = Field(..., alias="Shares Traded")
    Turnover_Cr: float = Field(..., alias="Turnover (₹ Cr)")


class VixRowInput(BaseModel):
    Date: str
    Open: float
    High: float
    Low: float
    Close: float
    Prev_Close: float = Field(..., alias="Prev. Close")

# ======================================================
# POST /feed/nifty
# ======================================================
@app.post("/feed/nifty")
def feed_nifty(rows: List[NiftyRow]):
    if not rows:
        raise HTTPException(status_code=400, detail="No rows provided")

    with open(NIFTY_CSV_FILE, "a+", newline="", encoding="utf-8") as f:
        ensure_newline_before_append(f)
        writer = csv.writer(f)

        for row in rows:
            validate_date(row.Date)
            writer.writerow([
                row.Date,
                row.Open,
                row.High,
                row.Low,
                row.Close,
                row.Shares_Traded,
                row.Turnover_Cr
            ])

    return {
        "status": "success",
        "rows_inserted": len(rows)
    }

# ======================================================
# POST /feed/vix
# ======================================================
@app.post("/feed/vix")
def feed_vix(rows: List[VixRowInput]):
    if not rows:
        raise HTTPException(status_code=400, detail="No rows provided")

    with open(VIX_CSV_FILE, "a+", newline="", encoding="utf-8") as f:
        ensure_newline_before_append(f)
        writer = csv.writer(f)

        for row in rows:
            validate_date(row.Date)

            change = round(row.Close - row.Prev_Close, 4)
            pct_change = round((change / row.Prev_Close) * 100, 2)

            writer.writerow([
                row.Date,
                row.Open,
                row.High,
                row.Low,
                row.Close,
                row.Prev_Close,
                change,
                pct_change
            ])

    return {
        "status": "success",
        "rows_inserted": len(rows)
    }


@app.get("/data/nifty")
def get_nifty_data(days: int, from_: str = "end"):
    data = read_csv_slice(
        file_path=NIFTY_CSV_FILE,
        days=days,
        direction=from_
    )

    return {
        "symbol": "NIFTY",
        "direction": from_,
        "days": days,
        "rows_returned": len(data),
        "data": data
    }


@app.get("/data/vix")
def get_vix_data(days: int, from_: str = "end"):
    data = read_csv_slice(
        file_path=VIX_CSV_FILE,
        days=days,
        direction=from_
    )

    return {
        "symbol": "INDIA_VIX",
        "direction": from_,
        "days": days,
        "rows_returned": len(data),
        "data": data
    }

# ======================================================
# /predict ENDPOINT
# ======================================================



@app.get("/predict")
def predict():
    return run_prediction()