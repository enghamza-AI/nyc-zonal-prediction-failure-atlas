# data.py
# Atolus — NYC Zonal Prediction Failure Atlas
# Loads, cleans, and splits NYC property sales by borough

import pandas as pd
import numpy as np
from pathlib import Path

# Borough codes in NYC government data
BOROUGH_NAMES = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "Staten Island"
}

DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "nyc_sales.csv"


def load_data():
    df = pd.read_csv(RAW_FILE, low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def clean_data(df):
    cols = [
        "borough", "sale_price", "gross_square_feet",
        "land_square_feet", "year_built", "total_units"
    ]
    df = df[cols].copy()

    # Remove $ and commas from sale_price before converting
    df["sale_price"] = df["sale_price"].astype(str).str.replace(",", "").str.replace("$", "").str.strip()

    numeric_cols = [
        "sale_price", "gross_square_feet",
        "land_square_feet", "year_built", "total_units"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["sale_price", "gross_square_feet", "year_built", "land_square_feet", "total_units"])
    df = df[df["sale_price"] >= 100000]
    df = df[df["gross_square_feet"] > 0]
    df = df[df["year_built"] >= 1800]

    return df


def split_by_borough(df):
    df["borough"] = df["borough"].map(BOROUGH_NAMES)

    boroughs = {}
    for name in BOROUGH_NAMES.values():
        borough_df = df[df["borough"] == name].copy()
        boroughs[name] = borough_df
        print(f"{name}: {len(borough_df):,} transactions")

    return boroughs


def run():
    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)
    print(f"Clean transactions: {len(df):,}")

    print("\nSplitting by borough...")
    boroughs = split_by_borough(df)

    return boroughs


if __name__ == "__main__":
    boroughs = run()