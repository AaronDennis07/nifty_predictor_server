from fastapi import HTTPException
import csv
import os

def read_csv_slice(file_path: str, days: int, direction: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="CSV file not found")

    if days <= 0:
        raise HTTPException(status_code=400, detail="days must be > 0")

    if direction not in ("start", "end"):
        raise HTTPException(status_code=400, detail="from must be 'start' or 'end'")

    with open(file_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        return []

    if direction == "start":
        return reader[:days]
    else:
        return reader[-days:]
