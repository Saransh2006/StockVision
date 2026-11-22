# 📈 StockVision — Interactive Stock Market Analyzer & ML Dashboard

StockVision is a Python-based financial analytics and machine learning dashboard built with **Streamlit** and **Plotly**. It fetches historical OHLCV market data from Yahoo Finance, computes technical indicators and risk metrics, compares stocks against benchmarks, and runs a Random Forest Regression model to estimate the next trading session's closing price.

---

## ✨ Features

* **Market Overview:** Interactive candlestick charts with SMA 20/50 overlays, daily trading volume, and session statistics.
* **Technical Analysis:** Moving Averages (SMA 20, SMA 50, EMA 20), Relative Strength Index (RSI 14 with overbought/oversold zones), and MACD (12, 26, 9) with histogram.
* **Risk & Comparison:** Annualized volatility, historical maximum drawdown curve, normalized performance comparison (Base = 100), and daily-return correlation against benchmarks like `^NSEI` (NIFTY 50).
* **Machine Learning Forecast:** Random Forest Regression (`250` trees) trained on 8 engineered features using a chronological 80/20 train-test split, complete with MAE, RMSE, R², actual-vs-predicted plots, and feature importance.
* **Dataset & Export:** View the processed historical dataset with all calculated indicators and download it as a `.csv` file.
* **Fintech Terminal UI:** Custom dark-mode interface featuring glassmorphic KPI cards, segmented pill tabs, and monospace financial typography.

---

## 🛠️ Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.11+** | Core application logic |
| **Streamlit** | Interactive web dashboard framework |
| **Plotly** | Interactive financial & statistical charts |
| **yfinance** | Historical stock market data retrieval (Yahoo Finance) |
| **Pandas & NumPy** | Data cleaning, preprocessing, and financial math |
| **Scikit-learn** | Random Forest Regression and evaluation metrics |

---

## 📂 Project Structure

```text
StockVision/
├── app.py              # Main Streamlit application & UI
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .venv/              # Local virtual environment (created during setup)
