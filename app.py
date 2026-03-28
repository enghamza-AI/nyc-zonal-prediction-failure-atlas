# app.py
# Atolus — NYC Zonal Prediction Failure Atlas
# Streamlit interactive dashboard

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import geopandas as gpd
from decompose import run as get_decomposition

GEO_FILE = "data/nyc_boroughs.gpkg"

st.set_page_config(
    page_title="Atolus — NYC Zonal Prediction Failure Atlas",
    page_icon="🗽",
    layout="wide"
)


@st.cache_data
def load_results():
    return get_decomposition()


def draw_bar_chart(results, borough):
    data = results[borough]
    bias = data["lr"]["bias_squared"] / 1e9
    variance = data["lr"]["variance"] / 1e9

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Bias²", "Variance"],
        [bias, variance],
        color=["#E74C3C", "#3498DB"],
        alpha=0.85,
        width=0.4
    )

    ax.set_ylabel("Error (Billions $²)")
    ax.set_title(f"{borough} — Bias vs Variance\nLinear Regression")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, [bias, variance]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:,.1f}B",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    return fig


def draw_map(results):
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
    gdf["bias_ratio"] = gdf["bias_squared"] / (
        gdf["bias_squared"] + gdf["variance"]
    )

    fig, ax = plt.subplots(figsize=(10, 7))

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
        ax.annotate(
            f"{row['borough']}\n{row['dominant']}",
            xy=(centroid.x, centroid.y),
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_title(
        "Atolus — NYC Zonal Prediction Failure Atlas\n"
        "Red = High Bias | Green = High Variance",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# ── MAIN DASHBOARD ──

st.title("Atolus — NYC Zonal Prediction Failure Atlas")
st.markdown(
    "Mapping where ML models break — zone by zone, "
    "across 27,000 real NYC property transactions."
)

with st.spinner("Running bootstrap decomposition across 5 boroughs..."):
    results = load_results()

st.success("Decomposition complete.")

# ── ROW 1: SUMMARY METRICS ──
st.subheader("Borough Summary")
cols = st.columns(5)
for col, borough in zip(cols, results.keys()):
    data = results[borough]
    bias = data["lr"]["bias_squared"] / 1e9
    var = data["lr"]["variance"] / 1e9
    dominant = data["dominant"]
    color = "HIGH BIAS" if dominant == "HIGH BIAS" else "HIGH VARIANCE"
    col.metric(
        label=borough,
        value=dominant,
        delta=f"Bias: {bias:,.0f}B | Var: {var:,.0f}B"
    )

st.divider()

# ── ROW 2: MAP + BOROUGH DETAIL ──
col_map, col_detail = st.columns([1.5, 1])

with col_map:
    st.subheader("Failure Atlas Map")
    st.pyplot(draw_map(results))

with col_detail:
    st.subheader("Borough Deep Dive")
    selected = st.selectbox(
        "Select a borough to inspect:",
        list(results.keys())
    )

    data = results[selected]
    lr = data["lr"]
    dt = data["dt"]

    st.pyplot(draw_bar_chart(results, selected))

    st.markdown("**Linear Regression**")
    st.write(f"Bias²: {lr['bias_squared']/1e9:,.1f}B")
    st.write(f"Variance: {lr['variance']/1e9:,.1f}B")
    st.write(f"Total Error: {lr['total_error']/1e9:,.1f}B")

    st.markdown("**Decision Tree**")
    st.write(f"Bias²: {dt['bias_squared']/1e9:,.1f}B")
    st.write(f"Variance: {dt['variance']/1e9:,.1f}B")
    st.write(f"Total Error: {dt['total_error']/1e9:,.1f}B")

    st.markdown("**Dominant Failure Mode**")
    if data["dominant"] == "HIGH BIAS":
        st.error("HIGH BIAS — Model too simple for this zone")
    else:
        st.success("HIGH VARIANCE — Model overfitting this zone")

st.divider()

# ── ROW 3: ALL BOROUGHS BAR CHART ──
st.subheader("All Boroughs — Bias vs Variance Comparison")

boroughs = list(results.keys())
bias_vals = [results[b]["lr"]["bias_squared"] / 1e9 for b in boroughs]
var_vals = [results[b]["lr"]["variance"] / 1e9 for b in boroughs]

x = np.arange(len(boroughs))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width/2, bias_vals, width,
       label="Bias²", color="#E74C3C", alpha=0.85)
ax.bar(x + width/2, var_vals, width,
       label="Variance", color="#3498DB", alpha=0.85)

ax.set_xlabel("Borough", fontsize=12)
ax.set_ylabel("Error (Billions $²)", fontsize=12)
ax.set_title("Atolus — All Boroughs Bias vs Variance\nLinear Regression",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(boroughs, rotation=15)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.divider()
st.caption(
    "Atolus — Stage 1 Week 2 | NYC OpenData 760k transactions | "
    "Bootstrap N=50 | Linear Regression + Decision Tree"
)