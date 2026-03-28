# visualize.py
# Atolus — NYC Zonal Prediction Failure Atlas
# Geographic and statistical visualization of bias-variance results

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from decompose import run as get_decomposition

GEO_FILE = "data/nyc_boroughs.gpkg"
OUTPUT_BAR = "outputs/bias_variance_bars.png"
OUTPUT_MAP = "outputs/failure_atlas_map.png"


def plot_bias_variance_bars(results):
    import os
    os.makedirs("outputs", exist_ok=True)

    boroughs = list(results.keys())
    bias_vals = [results[b]["lr"]["bias_squared"] for b in boroughs]
    var_vals = [results[b]["lr"]["variance"] for b in boroughs]

    bias_vals = [b / 1e9 for b in bias_vals]
    var_vals = [v / 1e9 for v in var_vals]

    x = np.arange(len(boroughs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, bias_vals, width,
                   label="Bias²", color="#E74C3C", alpha=0.85)
    bars2 = ax.bar(x + width/2, var_vals, width,
                   label="Variance", color="#3498DB", alpha=0.85)

    ax.set_xlabel("Borough", fontsize=12)
    ax.set_ylabel("Error (Billions $²)", fontsize=12)
    ax.set_title("Atolus — Bias vs Variance by NYC Borough\nLinear Regression",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(boroughs, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_BAR, dpi=150)
    print(f"Bar chart saved to {OUTPUT_BAR}")
    plt.close()


def plot_failure_map(results):
    gdf = gpd.read_file(GEO_FILE)

    gdf["bias_squared"] = gdf["borough"].map(
        {b: results[b]["lr"]["bias_squared"] / 1e9 for b in results}
    )
    gdf["variance"] = gdf["borough"].map(
        {b: results[b]["lr"]["variance"] / 1e9 for b in results}
    )
    gdf["dominant"] = gdf["borough"].map(
        {b: results[b]["dominant"] for b in results}
    )
    gdf["bias_ratio"] = gdf["bias_squared"] / (gdf["bias_squared"] + gdf["variance"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    gdf.plot(
        column="bias_ratio",
        ax=ax,
        cmap="RdYlGn_r",
        legend=True,
        legend_kwds={
            "label": "Bias Ratio (1.0 = Pure Bias, 0.0 = Pure Variance)",
            "orientation": "horizontal"
        }
    )

    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        dominant = row["dominant"]
        name = row["borough"]
        ax.annotate(
            f"{name}\n{dominant}",
            xy=(centroid.x, centroid.y),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="black"
        )

    ax.set_title(
        "Atolus — NYC Zonal Prediction Failure Atlas\nRed = High Bias | Green = High Variance",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(OUTPUT_MAP, dpi=150)
    print(f"Map saved to {OUTPUT_MAP}")
    plt.close()


def run():
    print("Running decomposition...")
    results = get_decomposition()

    print("\nGenerating bar chart...")
    plot_bias_variance_bars(results)

    print("Generating failure map...")
    plot_failure_map(results)

    print("\nDone. Check your outputs/ folder.")
    return results


if __name__ == "__main__":
    run()