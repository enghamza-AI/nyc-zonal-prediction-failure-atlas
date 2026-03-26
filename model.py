# model.py
# Atolus - NYC Zonal Prediction Failure Atlas
# Train Linear Regression + Decision Tree Per borough

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data import run as load_boroughs

FEATURES = ["gross_square_feet", "land_square_feet", "year_built", "total_units"]
TARGET = "sale_price"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def train_borough(borough_name, df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_preds)

    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=None, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_mse = mean_squared_error(y_test, dt_preds)

    print(f"\n{borough_name}:")
    print(f"  Linear Regression MSE : {lr_mse:,.0f}")
    print(f" Decision Tree MSE      : {dt_mse:,.0f} ")


    return {
        "borough": borough_name,
        "X_test": X_test,
        "y_test": y_test,
        "lr_model": lr,
        "dt_model": dt,
        "lr_preds": lr_preds,
        "dt_preds": dt_preds,
        "lr_mse": lr_mse,
        "dt_mse": dt_mse,
        "X_train": X_train,
        "y_train": y_train
    }

def run():
    print("Loading borough data...")
    boroughs = load_boroughs()

    print("\nTraining models per borough...")
    results = {}
    for borough_name, df in boroughs.items():
        results[borough_name] = train_borough(borough_name, df)

    return results
    
if __name__ == "__main__":
    results = run()
    print("\nModel training complete.")
    print(f"Boroughs trained: {list(results.keys())}")
