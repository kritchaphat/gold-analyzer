import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
import pandas as pd
import yfinance as yf

# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("gold-analyzer")

# ใช้ GC=F (Gold Futures) เพราะเสถียรกว่า XAUUSD=X ใน Yahoo Finance
SYMBOL = "GC=F" 
OUTPUT_DIR = Path("output")
RESULT_FILE = OUTPUT_DIR / "analysis_result.txt"

TIMEFRAMES = {
    "1H": {"interval": "60m", "period": "730d", "resample": None},
    "30M": {"interval": "30m", "period": "60d", "resample": None},
    "15M": {"interval": "15m", "period": "60d", "resample": None},
    "5M": {"interval": "5m", "period": "60d", "resample": None},
    "4H": {"interval": "60m", "period": "730d", "resample": "4h"},
}

@dataclass
class TimeframeResult:
    name: str
    df: pd.DataFrame

def safe_download(symbol: str, interval: str, period: str, retries: int = 5) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading {symbol} ({interval}), attempt {attempt}/{retries}")
            time.sleep(2)
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=interval, period=period, auto_adjust=False)
            if df.empty: raise ValueError("No data returned")
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception as e:
            if attempt == retries: raise e
            time.sleep(5)

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # EMAs
    out["EMA_9"] = out["Close"].ewm(span=9, adjust=False).mean()
    out["EMA_21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    out["RSI_14"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_S"] = out["MACD"].ewm(span=9, adjust=False).mean()
    return out.dropna()

def call_claude(results: List[TimeframeResult]) -> str:
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key: return "Error: No API Key"
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # เตรียมข้อมูลตัวเลขสรุป
    data_summary = ""
    for r in results:
        last = r.df.iloc[-1]
        prev = r.df.iloc[-2]
        trend = "UP" if last['Close'] > last['EMA_21'] else "DOWN"
        
        data_summary += f"""
[Timeframe: {r.name}]
- Price: Close={last['Close']:.2f}, Open={last['Open']:.2f}, High={last['High']:.2f}, Low={last['Low']:.2f}
- Indicators: EMA9={last['EMA_9']:.2f}, EMA21={last['EMA_21']:.2f}, EMA50={last['EMA_50']:.2f}
- Momentum: RSI={last['RSI_14']:.2f}, MACD={last['MACD']:.4f}, Signal={last['MACD_S']:.4f}
- Trend Signal: {trend}
"""

    prompt = f"""Act as an expert XAU/USD (Gold) trader. Analyze the following multi-timeframe technical data and provide:
    1. Market Sentiment & Trend Analysis
    2. Key Support & Resistance zones
    3. Trade Recommendation (Buy/Sell/Wait)
    4. Entry, Stop Loss, and Take Profit levels
    
    Data:
    {data_summary}
    """

    try:
        res = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content[0].text
    except Exception as e:
        return f"AI Analysis Failed: {str(e)}"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    
    # ดึงข้อมูลและคำนวณทีละ Timeframe
    for tf, cfg in TIMEFRAMES.items():
        try:
            df = safe_download(SYMBOL, cfg["interval"], cfg["period"])
            if cfg["resample"]: df = resample_ohlcv(df, cfg["resample"])
            df = add_indicators(df)
            results.append(TimeframeResult(tf, df))
            logger.info(f"Successfully processed {tf}")
        except Exception as e:
            logger.error(f"Failed {tf}: {e}")

    if results:
        analysis = call_claude(results)
        print("\n=== CLAUDE GOLD ANALYSIS ===\n")
        print(analysis)
        RESULT_FILE.write_text(analysis, encoding="utf-8")
    else:
        logger.error("No data available to analyze.")

if __name__ == "__main__":
    main()
