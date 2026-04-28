import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("gold-analyzer")

SYMBOL = "GC=F" 
OUTPUT_DIR = Path("output")
CHART_DIR = OUTPUT_DIR / "charts"
RESULT_FILE = OUTPUT_DIR / "analysis_result.txt"

TIMEFRAMES = {
    "1H": {"interval": "60m", "period": "730d", "resample": None},
    "30M": {"interval": "30m", "period": "60d", "resample": None},
    "15M": {"interval": "15m", "period": "60d", "resample": None},
    "5M": {"interval": "5m", "period": "60d", "resample": None},
    "4H": {"interval": "60m", "period": "730d", "resample": "4h"}, # ปรับ H เป็น h ตามคำแนะนำของระบบ
}

@dataclass
class TimeframeResult:
    name: str
    df: pd.DataFrame
    chart_path: Path

def safe_download(symbol: str, interval: str, period: str, retries: int = 5) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading {symbol} ({interval}), attempt {attempt}/{retries}")
            time.sleep(2)
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=interval, period=period, auto_adjust=False)
            if df.empty: raise ValueError("No data")
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception as e:
            if attempt == retries: raise e
            time.sleep(5)

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA_9"] = out["Close"].ewm(span=9, adjust=False).mean()
    out["EMA_21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()
    
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI_14"] = 100 - (100 / (1 + rs))
    
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_S"] = out["MACD"].ewm(span=9, adjust=False).mean()
    return out.dropna()

def make_chart(df: pd.DataFrame, timeframe: str, out_path: Path) -> None:
    # --- จุดที่แก้: ตัดข้อมูล 100 แท่งล่าสุดมาก่อน แล้วค่อยสร้าง plots ---
    plot_df = df.tail(100) 
    
    style = mpf.make_mpf_style(base_mpf_style="yahoo", facecolor="#0e1117", figcolor="#0e1117", gridstyle="--")
    
    ap = [
        mpf.make_addplot(plot_df["EMA_9"], color="#22c55e"),
        mpf.make_addplot(plot_df["EMA_21"], color="#3b82f6"),
        mpf.make_addplot(plot_df["EMA_50"], color="#f59e0b"),
        mpf.make_addplot(plot_df["RSI_14"], panel=1, color="#06b6d4", ylim=(0, 100)),
        mpf.make_addplot(plot_df["MACD"], panel=2, color="#60a5fa"),
        mpf.make_addplot(plot_df["MACD_S"], panel=2, color="#f97316"),
    ]
    
    mpf.plot(plot_df, type="candle", style=style, addplot=ap, title=f"GOLD - {timeframe}", savefig=str(out_path), volume=False)

def call_claude(results: List[TimeframeResult]) -> str:
    api_key = os.environ.get("CLAUDE_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    content = [{"type": "text", "text": "Analyze these gold charts for current trade setup."}]
    for r in results:
        with open(r.chart_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        content.append({"type": "text", "text": f"Timeframe: {r.name}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_data}})
    
    res = client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=1000, messages=[{"role": "user", "content": content}])
    return res.content[0].text

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for tf, cfg in TIMEFRAMES.items():
        df = safe_download(SYMBOL, cfg["interval"], cfg["period"])
        if cfg["resample"]: df = resample_ohlcv(df, cfg["resample"])
        df = add_indicators(df)
        path = CHART_DIR / f"{tf}.png"
        make_chart(df, tf, path)
        results.append(TimeframeResult(tf, df, path))
    
    analysis = call_claude(results)
    print(analysis)
    RESULT_FILE.write_text(analysis)

if __name__ == "__main__":
    main()
