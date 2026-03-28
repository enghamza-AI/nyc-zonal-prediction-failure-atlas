# Atolus — NYC Zonal Prediction Failure Atlas

> Mapping where ML models break — zone by zone, across 800k real NYC property transactions.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/enghamza-AI/Atolus)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/enghamza-AI/nyc-zonal-prediction-failure-atlas)

---

## What Is Atolus?

Most ML projects report a single accuracy number. Atolus asks a harder question:

**Does a model fail differently depending on where it's deployed?**

Atolus trains Linear Regression and Decision Tree models on 27,000 real NYC 
property transactions across 5 geographic zones — Manhattan, Bronx, Brooklyn, 
Queens, and Staten Island — then performs bootstrap bias-variance decomposition 
per zone to build a geographic failure atlas.

The result: a map of NYC where each borough is colored by its dominant ML 
failure mode. Not just "the model has X% error" — but where it fails, 
why it fails, and what kind of failure dominates each zone.

---

## The Real World Problem

A model trained on Manhattan data fails in the Bronx. Not because it is broken 
— because it learned Manhattan's logic. Luxury micro-neighborhoods. Glass towers. 
$3,000 per square foot. That logic does not transfer to Bronx multi-family 
housing or Staten Island suburbs.

This is geographic bias-variance shift — one of the most important and least 
discussed problems in deployed ML systems. Atolus makes it visible.

---

## What I Built

| File | Purpose |
|---|---|
| `data.py` | Loads 760k NYC property sales, cleans, splits by borough |
| `model.py` | Trains Linear Regression + Decision Tree per zone |
| `decompose.py` | Bootstrap bias-variance decomposition (N=50) per zone |
| `visualize.py` | Geographic failure map + bias-variance bar charts |
| `app.py` | Interactive Streamlit dashboard |

---

## Key Findings

| Borough | Dominant Failure | Why |
|---|---|---|
| Manhattan | HIGH BIAS | Extreme price range, 4 features insufficient |
| Bronx | HIGH BIAS | Complex mixed housing, model oversimplifies |
| Brooklyn | HIGH BIAS | Gentrifying rapidly, patterns unstable |
| Queens | HIGH BIAS | Large diverse market, linear model struggles |
| Staten Island | Most Stable | Suburban consistency, simplest patterns |

Linear Regression underfits every borough — Brooklyn catastrophically (Bias 46x > Variance).
Decision Tree begins overfitting in Queens and Bronx where variance exceeds bias.

---

## Technical Details

**Dataset:** NYC Citywide Annualized Calendar Sales — 760,914 transactions, 
filtered to 27,121 clean residential sales above $100,000

**Features:** Gross Square Feet, Land Square Feet, Year Built, Total Units

**Models:** Linear Regression, Decision Tree (unbounded depth)

**Decomposition Method:** Bootstrap sampling N=50, fixed test set per borough

**Bias Formula:**
```
Bias² = mean((mean_predictions - true_prices)²)
```

**Variance Formula:**
```
Variance = mean(std(50_predictions)²)
```

---

## Why This Matters

Geographic bias-variance analysis is what ML research papers do. 
Doing it on 800k real government transactions — not a toy dataset — 
and deploying it as an interactive atlas is rare for any self-learner 
at any level.

This project is part of a broader AI systems engineering roadmap 
targeting production-grade ML skills for roles at frontier AI labs.

---

## How to Run Locally
```bash
git clone https://github.com/enghamza-AI/nyc-zonal-prediction-failure-atlas
cd nyc-zonal-prediction-failure-atlas
pip install -r requirements.txt
```

Download NYC property sales data from NYC OpenData and place as `data/nyc_sales.csv`
```bash
python save_results.py   # pre-compute bootstrap results
streamlit run app.py     # launch dashboard
```

---

## Live Demo

[Launch Atolus on HuggingFace Spaces](https://huggingface.co/spaces/enghamza-AI/Atolus)

---

## Part of the Diamond AI Roadmap

**Stage 1 Week 2** of an 11-stage AI systems engineering self-study.

Built by Hamza — BSAI student, self-studying AI systems engineering 
targeting Anthropic, xAI, OpenAI, Perplexity, and YC-backed startups.

Previous project: [Vexis](https://huggingface.co/spaces/enghamza-AI/vexis) 
— Clinical AI Corruption Diagnostic Tool on MIMIC-III data.
