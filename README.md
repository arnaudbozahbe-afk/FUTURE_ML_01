# 🛒 Store Sales Time Series Forecasting

> Predicting daily sales per product family and store for a large Ecuadorian grocery chain using machine learning and advanced feature engineering.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-gradient_boosting-green?style=flat-square)](https://lightgbm.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Store_Sales_Competition-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
[![Metric](https://img.shields.io/badge/Metric-RMSLE-orange?style=flat-square)]()

---

## 📌 Overview

This project tackles the **Kaggle Store Sales Time Series Forecasting** competition. The goal is to accurately forecast daily sales for **54 stores** across **33 product families** of Corporación Favorita, a large grocery retailer in Ecuador.

The challenge lies in capturing complex patterns: **seasonality**, **holiday effects**, **oil price fluctuations** (Ecuador is oil-dependent), and **promotion impacts** — all simultaneously.

---

## 📁 Project Structure

```
store-sales-forecasting/
│
├── store_sales_forecasting.ipynb   # Main notebook (EDA → Features → Model → Submission)
├── data/
│   ├── train.csv                   # Historical sales (2013–2017)
│   ├── test.csv                    # Target prediction period
│   ├── stores.csv                  # Store metadata (city, state, type, cluster)
│   ├── oil.csv                     # Daily oil prices (WTI)
│   ├── holidays_events.csv         # National/regional/local holidays
│   └── transactions.csv            # Daily store transactions
└── submission.csv                  # Final Kaggle submission
```

---

## 🔍 Dataset

| File | Description |
|---|---|
| `train.csv` | ~3M rows — date, store, family, sales, onpromotion |
| `test.csv` | 28-day forecast window per store/family |
| `stores.csv` | Store city, state, type, cluster |
| `oil.csv` | WTI crude oil price (interpolated) |
| `holidays_events.csv` | National / Regional / Local holidays + transfers |
| `transactions.csv` | Daily transaction counts per store |

**Key stats:**
- 📅 Training period: January 2013 → August 2017
- 🏪 54 stores across Ecuador
- 🛍️ 33 product families
- 0️⃣ ~60% of sales are zero (sparse data challenge)

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis
- Total daily & monthly sales trends
- Sales distribution analysis → justifies `log1p` transformation
- Top 15 product families by volume
- Oil price vs. sales correlation
- Day-of-week seasonality patterns

### 2. Feature Engineering

| Feature Group | Features |
|---|---|
| **Time** | year, month, day, day_of_week, ISO week, quarter, is_weekend, month_start/end |
| **Holidays** | national / regional / local holiday counts, transfer flag |
| **Oil** | interpolated WTI price, 7-day moving average |
| **Sales Lags** | lag-1, lag-7, lag-14, lag-28 |
| **Rolling Stats** | 7/14/28-day rolling mean, 7-day rolling std |
| **Promo Lags** | `onpromotion` lag-1, lag-7, lag-14 |
| **Store Context** | city, state, store type, cluster |

### 3. Modeling

**Primary model: LightGBM** (gradient boosted trees — state-of-the-art for tabular time series)  
**Fallback: Ridge Regression** (regularized linear model, if LightGBM unavailable)

```
Target: log1p(sales)   →  reduces skew, handles outliers, aligns with RMSLE metric
```

**LightGBM hyperparameters:**
```python
params = {
    "objective":        "regression",
    "metric":           "rmse",
    "learning_rate":    0.05,
    "num_leaves":       128,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "lambda_l1":        0.1,
}
```

### 4. Validation Strategy

- **Time-based split**: last **60 days** of training data used for validation (no data leakage)
- Metric: **RMSLE** (Root Mean Squared Logarithmic Error)

| Score Range | Meaning |
|---|---|
| > 0.50 | Simple baseline |
| ~ 0.40 | Decent model |
| < 0.30 | Good model ✅ |
| < 0.20 | Top Kaggle leaderboard 🏆 |

---

## 📊 Results & Diagnostics

- ✅ Actual vs. Predicted daily sales curve
- ✅ Actual vs. Predicted scatter plot (log space)
- ✅ Residuals / error distribution
- ✅ Top 20 feature importances (LightGBM gain)
- ✅ Per-family RMSLE breakdown (identify hardest families)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-green?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install lightgbm pandas numpy scikit-learn matplotlib seaborn
```

### Data
Download the dataset from [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and place the CSV files in your `DATA_DIR`.

### Run
```bash
jupyter notebook store_sales_forecasting.ipynb
```

Update `DATA_DIR` in the Setup cell to point to your local data folder.

---

## 📈 Key Insights

- 📅 **Weekends** drive significantly higher average sales than weekdays
- 🛢️ **Oil price** shows a negative correlation with total sales (Ecuador's economy sensitivity)
- 🎉 **Holidays** (especially national ones) create sharp sales spikes
- 📦 **Sales lags** (lag-7 in particular) are among the most predictive features
- 🛒 **GROCERY I** is the dominant product family by total volume
