
import html
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="StockVision | Market Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom Fintech Terminal CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-base: #070B14;
    --bg-surface: #0E1525;
    --bg-elevated: #141D32;
    --border-subtle: rgba(148, 163, 184, 0.14);
    --border-glow: rgba(0, 210, 255, 0.35);
    --text-primary: #F8FAFC;
    --text-muted: #94A3B8;
    --accent-cyan: #00D2FF;
    --accent-green: #10B981;
    --accent-red: #F43F5E;
    --accent-amber: #F59E0B;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #101B33 0%, #070B14 45%, #05080F 100%);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

section[data-testid="stSidebar"] {
    background-color: #0A101D;
    border-right: 1px solid var(--border-subtle);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSlider label {
    color: var(--text-muted);
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Hero Banner */
.hero-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    padding: 1.25rem 1.5rem;
    background: linear-gradient(135deg, rgba(14, 21, 37, 0.9), rgba(20, 29, 50, 0.65));
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(12px);
}
.hero-title {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    background: linear-gradient(90deg, #FFFFFF 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 0.88rem;
    margin-top: 0.2rem;
}
.ticker-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.9rem;
    background: rgba(0, 210, 255, 0.08);
    border: 1px solid rgba(0, 210, 255, 0.28);
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent-cyan);
}
.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 8px var(--accent-green);
}

/* Custom KPI Grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.9rem;
    margin-bottom: 1.4rem;
}
@media (max-width: 1100px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.kpi-card {
    background: linear-gradient(180deg, rgba(18, 27, 46, 0.88) 0%, rgba(12, 18, 32, 0.92) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.05rem 1.15rem;
    transition: border-color 0.2s ease, transform 0.2s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 112px;
}
.kpi-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-2px);
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 0.45rem;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.kpi-pill {
    display: inline-block;
    margin-top: 0.5rem;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-weight: 600;
    width: fit-content;
}
.pill-pos { background: rgba(16, 185, 129, 0.14); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.28); }
.pill-neg { background: rgba(244, 63, 94, 0.14); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.28); }
.pill-neu { background: rgba(148, 163, 184, 0.12); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.2); }

/* Segmented Pill Tabs */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: rgba(14, 21, 37, 0.85);
    padding: 0.35rem;
    border-radius: 12px;
    border: 1px solid var(--border-subtle);
    width: fit-content;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    border-radius: 9px;
    padding: 0.45rem 1rem;
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.88rem;
    border: none;
    background: transparent;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #00D2FF 0%, #3B82F6 100%);
    color: #05080F !important;
    font-weight: 700;
}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
div[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none;
}

/* Panel & News Cards */
.stat-panel {
    background: rgba(14, 21, 37, 0.78);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 0.52rem 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    font-size: 0.88rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-k { color: var(--text-muted); }
.stat-v { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--text-primary); }

.news-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
}
@media (max-width: 900px) {
    .news-grid { grid-template-columns: 1fr; }
}
a.news-card, a.news-card:hover, a.news-card:visited {
    text-decoration: none !important;
    color: inherit !important;
}
.news-card {
    background: linear-gradient(180deg, rgba(18, 27, 46, 0.85) 0%, rgba(12, 18, 32, 0.92) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 0.65rem;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.news-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-2px);
}
.news-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.74rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.45rem;
}
.news-pub {
    color: var(--accent-cyan);
    font-weight: 600;
    text-transform: uppercase;
}
.news-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.38;
    margin: 0 0 0.45rem 0;
}
.news-summary {
    font-size: 0.84rem;
    color: var(--text-muted);
    line-height: 1.48;
    margin: 0;
}
.news-link {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--accent-cyan);
    margin-top: 0.25rem;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, rgba(0, 210, 255, 0.14), rgba(59, 130, 246, 0.14));
    border: 1px solid rgba(0, 210, 255, 0.35);
    color: #E2E8F0;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent-cyan);
    background: linear-gradient(135deg, rgba(0, 210, 255, 0.25), rgba(59, 130, 246, 0.25));
    color: #FFFFFF;
}

[data-testid="stMetric"] {
    background: rgba(14, 21, 37, 0.85);
    border: 1px solid var(--border-subtle);
    padding: 0.9rem 1.1rem;
    border-radius: 12px;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def format_volume(vol: float) -> str:
    if vol >= 1e9:
        return f"{vol / 1e9:.2f}B"
    if vol >= 1e6:
        return f"{vol / 1e6:.2f}M"
    if vol >= 1e3:
        return f"{vol / 1e3:.1f}K"
    return f"{vol:,.0f}"

def style_fig(fig: go.Figure, height: int = 420, title: str = None, yaxis_title: str = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=dict(text=title, font=dict(size=15, color="#E2E8F0", family="Plus Jakarta Sans")) if title else None,
        paper_bgcolor="rgba(14, 21, 37, 0.55)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(family="Plus Jakarta Sans", color="#94A3B8", size=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#334155", font=dict(family="JetBrains Mono", color="#F8FAFC")),
        margin=dict(l=16, r=16, t=42 if title else 24, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.08)", zeroline=False, showline=False),
        yaxis=dict(title=yaxis_title, gridcolor="rgba(148, 163, 184, 0.08)", zeroline=False, showline=False),
    )
    return fig

# ---------- Sidebar ----------
st.sidebar.markdown("### ⚙️ Terminal Controls")
default_tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX", "RELIANCE.NS", "TCS.NS", "INFY.NS", "^NSEI"]
ticker = st.sidebar.selectbox("Select stock", default_tickers, index=2)
custom_ticker = st.sidebar.text_input("Custom Yahoo Finance ticker", placeholder="e.g., AMZN, ZOMATO.NS")
if custom_ticker.strip():
    ticker = custom_ticker.strip().upper()

period = st.sidebar.selectbox("Timeframe", ["6mo", "1y", "2y", "5y"], index=2)
compare_ticker = st.sidebar.selectbox(
    "Benchmark / Compare", ["^NSEI", "AAPL", "MSFT", "NVDA", "RELIANCE.NS", "TCS.NS", "INFY.NS"],
    index=0
)
forecast_days = st.sidebar.slider("Future forecast (trading days)", min_value=5, max_value=30, value=14, step=1)
st.sidebar.caption("Tip: Add `.NS` for NSE India stocks (e.g., `RELIANCE.NS`).")
if st.sidebar.button("↻ Refresh Market Data", use_container_width=True):
    st.cache_data.clear()

# ---------- Data Functions ----------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_data(symbol: str, selected_period: str) -> pd.DataFrame:
    frame = yf.download(
        symbol,
        period=selected_period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
        threads=False
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).title() for c in frame.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in frame.columns for c in required):
        return pd.DataFrame()
    frame = frame[required].copy()
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(symbol: str, limit: int = 10) -> list[dict]:
    try:
        raw_news = yf.Ticker(symbol).news or []
    except Exception:
        return []

    articles = []
    for item in raw_news[:limit]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or item.get("title")
        if not title:
            continue

        provider = content.get("provider")
        publisher = (
            provider.get("displayName") if isinstance(provider, dict)
            else content.get("publisher") or item.get("publisher") or "Yahoo Finance"
        )

        canonical = content.get("canonicalUrl")
        click = content.get("clickThroughUrl")
        link = (
            (canonical.get("url") if isinstance(canonical, dict) else None)
            or (click.get("url") if isinstance(click, dict) else None)
            or content.get("link")
            or item.get("link")
            or "#"
        )

        summary = content.get("summary") or content.get("description") or ""
        if len(summary) > 180:
            summary = summary[:177].rsplit(" ", 1)[0] + "..."

        pub_str = ""
        pub_date = content.get("pubDate") or content.get("displayTime")
        pub_ts = item.get("providerPublishTime")
        if pub_date:
            try:
                dt = datetime.fromisoformat(str(pub_date).replace("Z", "+00:00"))
                pub_str = dt.strftime("%d %b %Y · %H:%M")
            except Exception:
                pub_str = str(pub_date)[:16]
        elif pub_ts:
            pub_str = datetime.fromtimestamp(pub_ts).strftime("%d %b %Y · %H:%M")

        articles.append({
            "title": title,
            "publisher": publisher,
            "link": link,
            "summary": summary,
            "published": pub_str
        })
    return articles

def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["Close"]
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Daily_Return"] = close.pct_change()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std() * np.sqrt(252)
    df["Drawdown"] = close / close.cummax() - 1
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["return_1d"] = df["Close"].pct_change(1)
    feat["return_5d"] = df["Close"].pct_change(5)
    feat["sma20_gap"] = df["Close"] / df["SMA_20"] - 1
    feat["sma50_gap"] = df["Close"] / df["SMA_50"] - 1
    feat["rsi"] = df["RSI_14"]
    feat["macd"] = df["MACD"]
    feat["volume_change"] = df["Volume"].pct_change()
    feat["volatility_20"] = df["Daily_Return"].rolling(20).std()
    return feat

def forecast_future_prices(df: pd.DataFrame, model: RandomForestRegressor, feature_cols: list[str], horizon: int) -> pd.DataFrame:
    sim = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    avg_vol = float(df["Volume"].tail(20).mean())
    daily_sigma = float(df["Daily_Return"].dropna().std())
    future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon)

    rows = []
    for step, f_date in enumerate(future_dates, start=1):
        sim_ind = add_indicators(sim)
        sim_feat = build_features(sim_ind).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        x_last = sim_feat[feature_cols].iloc[[-1]]

        pred_ret = float(model.predict(x_last)[0])
        pred_ret = float(np.clip(pred_ret, -0.08, 0.08))
        prev_close = float(sim["Close"].iloc[-1])
        next_close = prev_close * (1 + pred_ret)

        sim.loc[f_date] = [prev_close, max(prev_close, next_close), min(prev_close, next_close), next_close, avg_vol]
        band = next_close * daily_sigma * np.sqrt(step)
        rows.append({
            "Date": f_date,
            "Predicted Close": next_close,
            "Daily Change %": pred_ret * 100,
            "Upper Band (+1σ)": next_close + band,
            "Lower Band (-1σ)": max(0.0, next_close - band),
        })
    return pd.DataFrame(rows).set_index("Date")

with st.spinner(f"Fetching {ticker} market data..."):
    raw = fetch_data(ticker, period)

if raw.empty or len(raw) < 60:
    st.error("No usable data returned. Check the ticker symbol or try a different timeframe.")
    st.stop()

data = add_indicators(raw)
latest = data.iloc[-1]
previous = data["Close"].iloc[-2]
price_change = (latest["Close"] / previous - 1) * 100
period_return = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100
annual_vol = data["Daily_Return"].std() * np.sqrt(252) * 100
max_dd = data["Drawdown"].min() * 100
vol_ratio = (latest["Volume"] / data["Volume"].mean() - 1) * 100

# ---------- Hero Header ----------
st.markdown(f"""
<div class="hero-header">
    <div>
        <h1 class="hero-title">StockVision</h1>
        <div class="hero-sub">Market analytics, live news, technical indicators &amp; multi-day ML forecasting</div>
    </div>
    <div class="ticker-badge">
        <span class="pulse-dot"></span>
        {ticker} · {period.upper()} · {data.index[-1].strftime('%d %b %Y')}
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Custom KPI Cards ----------
chg_cls = "pill-pos" if price_change >= 0 else "pill-neg"
chg_arrow = "▲" if price_change >= 0 else "▼"
ret_cls = "pill-pos" if period_return >= 0 else "pill-neg"
vol_cls = "pill-pos" if vol_ratio >= 0 else "pill-neu"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Latest Close</div>
        <div class="kpi-value">{latest['Close']:,.2f}</div>
        <div class="kpi-pill {chg_cls}">{chg_arrow} {price_change:+.2f}% today</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Period Return ({period})</div>
        <div class="kpi-value">{period_return:+.2f}%</div>
        <div class="kpi-pill {ret_cls}">{"Bullish trend" if period_return >= 0 else "Bearish trend"}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Annualized Volatility</div>
        <div class="kpi-value">{annual_vol:.2f}%</div>
        <div class="kpi-pill pill-neu">{"High" if annual_vol > 35 else "Moderate" if annual_vol > 20 else "Low"} risk</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Max Drawdown</div>
        <div class="kpi-value">{max_dd:.2f}%</div>
        <div class="kpi-pill pill-neg">Peak-to-trough</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Latest Volume</div>
        <div class="kpi-value">{format_volume(latest['Volume'])}</div>
        <div class="kpi-pill {vol_cls}">{vol_ratio:+.0f}% vs avg</div>
    </div>
</div>
""", unsafe_allow_html=True)

overview_tab, technical_tab, risk_tab, ml_tab, news_tab, data_tab = st.tabs(
    ["Overview", "Technical Indicators", "Risk & Comparison", "ML Forecast", "Market News", "Dataset"]
)

# ---------- Overview ----------
with overview_tab:
    left, right = st.columns([2.35, 1], gap="medium")
    with left:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="OHLC",
            increasing_line_color="#10B981", decreasing_line_color="#F43F5E"
        ))
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20", line=dict(color="#00D2FF", width=1.6)))
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], name="SMA 50", line=dict(color="#F59E0B", width=1.6)))
        style_fig(fig, height=500, title=f"{ticker} Price Action & Moving Averages")
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(f"""
        <div class="stat-panel">
            <div class="kpi-label" style="margin-bottom:0.75rem;">Session Statistics</div>
            <div class="stat-row"><span class="stat-k">Period High</span><span class="stat-v">{data['High'].max():,.2f}</span></div>
            <div class="stat-row"><span class="stat-k">Period Low</span><span class="stat-v">{data['Low'].min():,.2f}</span></div>
            <div class="stat-row"><span class="stat-k">Average Close</span><span class="stat-v">{data['Close'].mean():,.2f}</span></div>
            <div class="stat-row"><span class="stat-k">Avg Daily Volume</span><span class="stat-v">{format_volume(data['Volume'].mean())}</span></div>
            <div class="stat-row"><span class="stat-k">Trading Sessions</span><span class="stat-v">{len(data):,}</span></div>
            <div class="stat-row"><span class="stat-k">Latest RSI (14)</span><span class="stat-v">{latest['RSI_14']:.1f}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)
        bar_colors = np.where(data["Close"] >= data["Open"], "rgba(16, 185, 129, 0.7)", "rgba(244, 63, 94, 0.7)")
        volume_fig = go.Figure(go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color=bar_colors))
        style_fig(volume_fig, height=230, title="Daily Volume")
        st.plotly_chart(volume_fig, use_container_width=True)

# ---------- Technical Indicators ----------
with technical_tab:
    ma_fig = go.Figure()
    ma_fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#F8FAFC", width=2)))
    ma_fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20", line=dict(color="#00D2FF", width=1.6)))
    ma_fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], name="SMA 50", line=dict(color="#F59E0B", width=1.6)))
    ma_fig.add_trace(go.Scatter(x=data.index, y=data["EMA_20"], name="EMA 20", line=dict(color="#A855F7", width=1.5, dash="dot")))
    style_fig(ma_fig, height=380, title="Moving Averages Comparison")
    st.plotly_chart(ma_fig, use_container_width=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI_14"], name="RSI (14)", line=dict(color="#00D2FF", width=2)))
        rsi_fig.add_hrect(y0=70, y1=100, fillcolor="rgba(244, 63, 94, 0.08)", line_width=0)
        rsi_fig.add_hrect(y0=0, y1=30, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0)
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="#F43F5E", annotation_text="Overbought (70)")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="#10B981", annotation_text="Oversold (30)")
        style_fig(rsi_fig, height=320, title="Relative Strength Index (RSI 14)")
        rsi_fig.update_yaxes(range=[0, 100])
        st.plotly_chart(rsi_fig, use_container_width=True)

    with c2:
        macd_hist = data["MACD"] - data["MACD_Signal"]
        hist_colors = np.where(macd_hist >= 0, "rgba(16, 185, 129, 0.45)", "rgba(244, 63, 94, 0.45)")
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Bar(x=data.index, y=macd_hist, name="Histogram", marker_color=hist_colors))
        macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="#00D2FF", width=1.8)))
        macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], name="Signal", line=dict(color="#F59E0B", width=1.6)))
        style_fig(macd_fig, height=320, title="MACD (12, 26, 9)")
        st.plotly_chart(macd_fig, use_container_width=True)

    rsi_value = latest["RSI_14"]
    if rsi_value >= 70:
        st.warning(f"RSI is {rsi_value:.1f} — momentum is in the traditional overbought zone (>70).")
    elif rsi_value <= 30:
        st.success(f"RSI is {rsi_value:.1f} — momentum is in the traditional oversold zone (<30).")
    else:
        st.info(f"RSI is {rsi_value:.1f} — momentum is within the neutral 30–70 range.")

# ---------- Risk & Comparison ----------
with risk_tab:
    r1, r2, r3 = st.columns(3)
    r1.metric("Annualized Volatility", f"{annual_vol:.2f}%")
    r2.metric("Maximum Drawdown", f"{max_dd:.2f}%")
    r3.metric("Average Daily Return", f"{data['Daily_Return'].mean() * 100:.3f}%")

    dd_fig = go.Figure(go.Scatter(
        x=data.index, y=data["Drawdown"] * 100, fill="tozeroy", name="Drawdown %",
        line=dict(color="#F43F5E", width=1.8), fillcolor="rgba(244, 63, 94, 0.16)"
    ))
    style_fig(dd_fig, height=300, title="Historical Drawdown (%)", yaxis_title="Drawdown %")
    st.plotly_chart(dd_fig, use_container_width=True)

    if compare_ticker == ticker:
        st.warning("Select a different comparison ticker in the sidebar.")
    else:
        compare = fetch_data(compare_ticker, period)
        if compare.empty:
            st.warning(f"Could not load comparison data for {compare_ticker}.")
        else:
            returns = pd.concat([
                data["Close"].pct_change().rename(ticker),
                compare["Close"].pct_change().rename(compare_ticker)
            ], axis=1).dropna()
            if len(returns) > 2:
                corr = returns[ticker].corr(returns[compare_ticker])
                st.metric(f"Daily-Return Correlation ({ticker} vs {compare_ticker})", f"{corr:.3f}")
                norm = pd.concat([
                    data["Close"].rename(ticker),
                    compare["Close"].rename(compare_ticker)
                ], axis=1).dropna()
                normalized = norm / norm.iloc[0] * 100
                comp_fig = go.Figure()
                comp_fig.add_trace(go.Scatter(x=normalized.index, y=normalized[ticker], name=ticker, line=dict(color="#00D2FF", width=2.2)))
                comp_fig.add_trace(go.Scatter(x=normalized.index, y=normalized[compare_ticker], name=compare_ticker, line=dict(color="#F59E0B", width=2.2)))
                style_fig(comp_fig, height=380, title="Normalized Performance (Base = 100)", yaxis_title="Growth of 100")
                st.plotly_chart(comp_fig, use_container_width=True)
                st.caption("Correlation measures how daily returns moved together during the selected period; it does not imply causation.")

# ---------- Machine Learning (Multi-Day Future Forecast) ----------
with ml_tab:
    st.caption(
        "Random Forest Regression trained on 8 technical features to predict next-day returns, "
        "evaluated on a chronological 80/20 split, then rolled forward recursively to project future prices."
    )

    features = build_features(data)
    supervised = features.copy()
    supervised["target_return"] = data["Close"].pct_change().shift(-1)
    supervised["base_close"] = data["Close"]
    supervised["actual_next_close"] = data["Close"].shift(-1)
    supervised["target_date"] = pd.Series(data.index, index=data.index).shift(-1)
    supervised = supervised.replace([np.inf, -np.inf], np.nan).dropna()

    if len(supervised) < 80:
        st.warning("Not enough clean observations for ML training. Choose a longer timeframe.")
    else:
        feature_cols = list(features.columns)
        X = supervised[feature_cols]
        y_ret = supervised["target_return"]
        split = int(len(supervised) * 0.8)

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test_ret = y_ret.iloc[:split], y_ret.iloc[split:]

        eval_model = RandomForestRegressor(n_estimators=250, max_depth=7, min_samples_leaf=3, random_state=42, n_jobs=-1)
        eval_model.fit(X_train, y_train)
        pred_test_ret = eval_model.predict(X_test)

        actual_test_price = supervised["actual_next_close"].iloc[split:].values
        pred_test_price = supervised["base_close"].iloc[split:].values * (1 + pred_test_ret)

        mae = mean_absolute_error(actual_test_price, pred_test_price)
        rmse = np.sqrt(mean_squared_error(actual_test_price, pred_test_price))
        r2 = r2_score(actual_test_price, pred_test_price)

        final_model = RandomForestRegressor(n_estimators=250, max_depth=7, min_samples_leaf=3, random_state=42, n_jobs=-1)
        final_model.fit(X, y_ret)

        forecast_df = forecast_future_prices(data, final_model, feature_cols, forecast_days)
        next_day_price = float(forecast_df["Predicted Close"].iloc[0])
        horizon_end_price = float(forecast_df["Predicted Close"].iloc[-1])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Next Session Est.", f"{next_day_price:,.2f}", f"{(next_day_price / latest['Close'] - 1) * 100:+.2f}%")
        m2.metric(f"{forecast_days}-Day Target Est.", f"{horizon_end_price:,.2f}", f"{(horizon_end_price / latest['Close'] - 1) * 100:+.2f}%")
        m3.metric("Test MAE / RMSE", f"{mae:,.2f}", f"RMSE {rmse:,.2f}", delta_color="off")
        m4.metric("Test R²", f"{r2:.3f}")

        recent = data["Close"].tail(60)
        fc_fig = go.Figure()
        fc_fig.add_trace(go.Scatter(
            x=recent.index, y=recent.values, name="Historical Close",
            line=dict(color="#00D2FF", width=2.2)
        ))
        bridge_x = [recent.index[-1]] + list(forecast_df.index)
        bridge_y = [recent.iloc[-1]] + list(forecast_df["Predicted Close"])
        fc_fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y, name=f"{forecast_days}d Forecast",
            line=dict(color="#F59E0B", width=2.4, dash="dash")
        ))
        fc_fig.add_trace(go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index[::-1]),
            y=list(forecast_df["Upper Band (+1σ)"]) + list(forecast_df["Lower Band (-1σ)"][::-1]),
            fill="toself", fillcolor="rgba(245, 158, 11, 0.14)",
            line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="±1σ Volatility Band"
        ))
        style_fig(fc_fig, height=400, title=f"{ticker} {forecast_days}-Trading-Day Future Price Projection", yaxis_title="Price")
        st.plotly_chart(fc_fig, use_container_width=True)

        c_left, c_right = st.columns([1.35, 1], gap="medium")
        with c_left:
            pred_frame = pd.DataFrame(
                {"Actual": actual_test_price, "Predicted": pred_test_price},
                index=pd.to_datetime(supervised["target_date"].iloc[split:].values)
            )
            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(x=pred_frame.index, y=pred_frame["Actual"], name="Actual", line=dict(color="#00D2FF", width=1.9)))
            pred_fig.add_trace(go.Scatter(x=pred_frame.index, y=pred_frame["Predicted"], name="Predicted", line=dict(color="#10B981", width=1.8, dash="dot")))
            style_fig(pred_fig, height=320, title="Held-Out Test Set: Actual vs Predicted")
            st.plotly_chart(pred_fig, use_container_width=True)

        with c_right:
            importance = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values()
            imp_fig = go.Figure(go.Bar(
                x=importance.values, y=importance.index, orientation="h",
                marker=dict(color=importance.values, colorscale="Tealgrn")
            ))
            style_fig(imp_fig, height=320, title="Feature Importance")
            st.plotly_chart(imp_fig, use_container_width=True)

        with st.expander(f"View {forecast_days}-day forecast table", expanded=False):
            display_fc = forecast_df.copy()
            display_fc.index = display_fc.index.strftime("%Y-%m-%d")
            st.dataframe(
                display_fc.style.format({
                    "Predicted Close": "{:,.2f}",
                    "Daily Change %": "{:+.2f}%",
                    "Upper Band (+1σ)": "{:,.2f}",
                    "Lower Band (-1σ)": "{:,.2f}",
                }),
                use_container_width=True
            )

        st.caption("Educational forecast only — not financial advice. Omits news, fundamentals, and macro events.")

# ---------- Market News ----------
with news_tab:
    with st.spinner(f"Loading latest {ticker} headlines..."):
        news_items = fetch_news(ticker, limit=10)

    if not news_items:
        st.info(f"No recent Yahoo Finance news articles were returned for {ticker}. Try a major ticker like NVDA, AAPL, or RELIANCE.NS.")
    else:
        cards_html = []
        for art in news_items:
            t = html.escape(art["title"])
            p = html.escape(art["publisher"])
            d = html.escape(art["published"])
            s = html.escape(art["summary"])
            u = html.escape(art["link"], quote=True)
            summary_block = f'<p class="news-summary">{s}</p>' if s else ""
            cards_html.append(
                f'<a class="news-card" href="{u}" target="_blank" rel="noopener noreferrer">'
                f'<div>'
                f'<div class="news-meta"><span class="news-pub">{p}</span><span>{d}</span></div>'
                f'<h4 class="news-title">{t}</h4>'
                f'{summary_block}'
                f'</div>'
                f'<span class="news-link">Read full article ↗</span>'
                f'</a>'
            )
        st.markdown(f'<div class="news-grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)

# ---------- Dataset ----------
with data_tab:
    st.caption(f"{len(data):,} daily observations · {data.index.min().date()} to {data.index.max().date()}")
    st.dataframe(data.tail(100), use_container_width=True, height=420)
    csv = data.to_csv(index_label="Date").encode("utf-8")
    st.download_button(
        "⬇ Download Analyzed Dataset (CSV)",
        data=csv,
        file_name=f"{ticker.replace('^', '')}_{period}_stockvision.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("StockVision · Educational financial analytics dashboard · Data & news sourced from Yahoo Finance via yfinance.")
