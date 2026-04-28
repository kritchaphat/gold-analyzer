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

# ตั้งค่า Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("gold-analyzer")

# --- จุดที่ 1: เปลี่ยน Symbol เป็น GC=F เพื่อความเสถียร ---
SYMBOL = "GC=F" 
OUTPUT_DIR = Path("output")
CHART_DIR = OUTPUT_DIR / "charts"
RESULT_FILE = OUTPUT_DIR / "analysis_result.txt"
INDICATOR_FILE = OUTPUT_DIR / "indicator_snapshot.json"

TIMEFRAMES = {
    "1H": {"interval": "60m", "period": "730d", "resample": None},
    "30M": {"interval": "30m", "period": "60d", "resample": None},
    "15M": {"interval": "15m", "period": "60d", "resample": None},
    "5M": {"interval": "5m", "period": "60d", "resample": None},
    "4H": {"interval": "60m", "period": "730d", "resample": "4H"},
}

@dataclass
class TimeframeResult:
    name: str
    df: pd.DataFrame
    chart_path: Path
    latest_indicators: Dict[str, float]

def safe_download(symbol: str, interval: str, period: str, retries: int = 5, sleep_seconds: int = 5) -> pd.DataFrame:
    """Download data from yfinance with headers and retries."""
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Attempting download {symbol} ({interval}), attempt {attempt}/{retries}")
            
            # --- จุดที่ 2: เพิ่มการหน่วงเวลาเพื่อไม่ให้โดนบล็อก ---
            time.sleep(sleep_seconds) 
            
            # ดึงข้อมูล
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                interval=interval,
                period=period,
                auto_adjust=False,
            )

            if df is None or df.empty:
                raise ValueError(f"No data returned for {symbol}")

            # ปรับแต่งคอลัมน์ให้รองรับ Multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            
            if len(df) < 50:
                raise ValueError(f"Insufficient data: {len(df)} rows")

            return df
        except Exception as exc:
            last_error = exc
            logger.warning(f"Download failed on attempt {attempt}: {exc}")
            if attempt < retries:
                time.sleep(sleep_seconds * 2)

    raise RuntimeError(f"Failed after {retries} attempts: {last_error}")

# --- ฟังก์ชันอื่นๆ (Resample, Indicators, Chart) คงเดิมตามโค้ดต้นฉบับของคุณ ---
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    return out

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA_9"] = out["Close"].ewm(span=9, adjust=False).mean()
    out["EMA_21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()
    basis = out["Close"].rolling(window=20).mean()
    dev = out["Close"].rolling(window=20).std(ddof=0)
    out["BB_UPPER"] = basis + (2 * dev)
    out["BB_MID"] = basis
    out["BB_LOWER"] = basis - (2 * dev)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI_14"] = 100 - (100 / (1 + rs))
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    return out.dropna()

def latest_values(df: pd.DataFrame) -> Dict[str, float]:
    row = df.iloc[-1]
    return {k: float(row[k.upper() if k in ["macd", "rsi_14"] else k.replace("_", "").upper() if "ema" in k else k.upper()]) for k in ["close", "ema_9", "ema_21", "ema_50", "bb_upper", "bb_mid", "bb_lower", "rsi_14", "macd", "macd_signal", "macd_hist"]}

def make_chart(df: pd.DataFrame, timeframe: str, out_path: Path) -> None:
    style = mpf.make_mpf_style(base_mpf_style="yahoo", facecolor="#0e1117", figcolor="#0e1117", gridstyle="--")
    plots = [
        mpf.make_addplot(df["EMA_9"], color="#22c55e"),
        mpf.make_addplot(df["EMA_21"], color="#3b82f6"),
        mpf.make_addplot(df["EMA_50"], color="#f59e0b"),
        mpf.make_addplot(df["RSI_14"], panel=1, color="#06b6d4", ylim=(0, 100)),
        mpf.make_addplot(df["MACD"], panel=2, color="#60a5fa"),
    ]
    mpf.plot(df.tail(100), type="candle", style=style, addplot=plots, title=f"GOLD - {timeframe}", savefig=str(out_path))

def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def call_claude(results: List[TimeframeResult]) -> str:
    api_key = os.environ.get("CLAUDE_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    content = [{"type": "text", "text": "Analyze these gold charts for London/NY session."}]
    for item in results:
        content.append({"type": "text", "text": f"TF: {item.name}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_to_base64(item.chart_path)}})
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": content}]
    )
    return response.content[0].text

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for tf in ["4H", "1H", "30M", "15M", "5M"]:
        cfg = TIMEFRAMES[tf]
        df = safe_download(SYMBOL, cfg["interval"], cfg["period"])
        if cfg["resample"]: df = resample_ohlcv(df, cfg["resample"])
        df = add_indicators(df)
        path = CHART_DIR / f"{tf}.png"
        make_chart(df, tf, path)
        results.append(TimeframeResult(tf, df, path, {}))
    
    analysis = call_claude(results)
    print(analysis)
    RESULT_FILE.write_text(analysis)
    return 0

if __name__ == "__main__":
    main()
