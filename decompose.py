# decompose.py
# Atolus - NYC zonal prediction failure atlas
# Bootstrap bias-variance decomposition per borough


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from data import run as load_boroughs

N_BOOTSTRAPS = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURES = ["gross_square_feet", "land_square_feet", "year_built", "total_units"]
TARGET = "sale_price"

def bootstrap_decompose(df, model_type="lr"):
    X = df[FEATURES].values
    y = df[TARGET].values

    n_test = int(len(X) * TEST_SIZE)
    X_test = X[:n_test]
    y_test = y[:n_test]
    X_pool = X[n_test:]
    y_pool = y[n_test:]

    all_predictions = np.zeros((N_BOOTSTRAPS, n_test))

    for i in range(N_BOOTSTRAPS):

        X_boot, y_boot = resample(X_pool, y_pool, random_state=i)

        if model_type == "lr":
            model = LinearRegression()
        else:
            model = DecisionTreeRegressor(max_depth=None, random_state=i)


        model.fit(X_boot, y_boot)
        all_predictions[i] = model.predict(X_test)


    mean_predictions = all_predictions.mean(axis=0)

    bias_squared = np.mean((mean_predictions - y_test) ** 2)
    variance = np.mean(all_predictions.var(axis=0))
    noise = 0
    total_error = bias_squared + variance

    return {
        "bias_squared": bias_squared,
        "variance": variance,
        "total_error": total_error
    }

def run():
    print("Loading borough data...")
    boroughs = load_boroughs()
    
    print("\nRunning bootstrap decomposition (50 runs per borough)...")
    results = {}
    
    for borough_name, df in boroughs.items():
        print(f"\n{borough_name}:")
        
        lr_results = bootstrap_decompose(df, model_type="lr")
        dt_results = bootstrap_decompose(df, model_type="dt")
        
        print(f"  LINEAR REGRESSION:")
        print(f"    Bias²    : {lr_results['bias_squared']:,.0f}")
        print(f"    Variance : {lr_results['variance']:,.0f}")
        print(f"    Total    : {lr_results['total_error']:,.0f}")
        
        print(f"  DECISION TREE:")
        print(f"    Bias²    : {dt_results['bias_squared']:,.0f}")
        print(f"    Variance : {dt_results['variance']:,.0f}")
        print(f"    Total    : {dt_results['total_error']:,.0f}")
        
        # Dominant failure mode
        if lr_results['bias_squared'] > lr_results['variance']:
            dominant = "HIGH BIAS"
        else:
            dominant = "HIGH VARIANCE"
            
        results[borough_name] = {
            "lr": lr_results,
            "dt": dt_results,
            "dominant": dominant
        }
        
        print(f"  → Dominant failure: {dominant}")
    
    return results


if __name__ == "__main__":
    results = run()
    print("\nDecomposition complete.")
