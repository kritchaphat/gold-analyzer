import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
import pandas as pd
import pandas_ta as ta  # หัวใจหลักในการคำนวณแบบ Kronos
import yfinance as yf

# --- 1. SETTINGS & LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("mini-kronos-gold")

SYMBOL = "GC=F"  # Gold Futures
OUTPUT_DIR = Path("output")
RESULT_FILE = OUTPUT_DIR / "analysis_result.txt"

# กำหนด Timeframes ที่ต้องการวิเคราะห์
TIMEFRAMES = {
    "4H": {"interval": "60m", "period": "730d", "resample": "4h"},
    "1H": {"interval": "60m", "period": "730d", "resample": None},
    "15M": {"interval": "15m", "period": "60d", "resample": None},
    "5M": {"interval": "5m", "period": "60d", "resample": None},
}

@dataclass
class TimeframeResult:
    name: str
    df: pd.DataFrame

# --- 2. DATA ENGINE ---
def safe_download(symbol: str, interval: str, period: str, retries: int = 5) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading {symbol} ({interval}), attempt {attempt}/{retries}")
            time.sleep(1)
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=interval, period=period, auto_adjust=False)
            if df.empty: raise ValueError("No data returned from Yahoo Finance")
            
            # จัดการ MultiIndex columns ถ้ามี
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
                
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception as e:
            if attempt == retries: raise e
            time.sleep(5)

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()

# --- 3. KRONOS INDICATORS ---
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    # ATR: ใช้เพื่อวัดความผันผวนและตั้ง SL/TP ตามแบบ Kronos
    out["ATR"] = out.ta.atr(length=14)
    
    # Trend: ใช้ EMA เพื่อดูประโยคของราคา
    out["EMA_21"] = out.ta.ema(length=21)
    out["EMA_50"] = out.ta.ema(length=50)
    
    # Momentum: RSI & MACD
    out["RSI_14"] = out.ta.rsi(length=14)
    macd = out.ta.macd(fast=12, slow=26, signal=9)
    out = pd.concat([out, macd], axis=1)
    
    return out.dropna()

# --- 4. MINI-KRONOS AI LOGIC ---
def call_claude(results: List[TimeframeResult]) -> str:
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key: return "Error: No CLAUDE_API_KEY found in environment."
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # สร้างข้อมูลสรุปแบบ Sequence (ภาษาแท่งเทียน) เพื่อส่งให้ AI
    data_summary = ""
    for r in results:
        last = r.df.iloc[-1]
        # ดึงราคาปิดย้อนหลัง 10 แท่ง เพื่อสร้างชุดข้อมูลแบบ Transformer
        sequence = r.df['Close'].tail(10).round(2).tolist()
        
        data_summary += f"""
[Timeframe: {r.name}]
- Candle Sequence (Last 10): {sequence}
- Current OHLC: O={last['Open']:.2f}, H={last['High']:.2f}, L={last['Low']:.2f}, C={last['Close']:.2f}
- Indicators: ATR={last['ATR']:.2f}, RSI={last['RSI_14']:.1f}, EMA21={last['EMA_21']:.2f}
- MACD: {last.get('MACD_12_26_9', 0):.4f} | Signal: {last.get('MACDs_12_26_9', 0):.4f}
"""

    # Prompt ที่ออกแบบมาเพื่อดึงพลังของ Foundation Model
    prompt = f"""You are 'Mini-Kronos', a specialized AI Foundation Model for Financial Markets.
Analyze the following Gold (GC=F) candle sequences as a linguistic pattern.

### INSTRUCTIONS:
1. Translate the candle sequences into a Market Bias (Bullish/Bearish).
2. Identify Liquidity Zones (BSL/SSL) and Fair Value Gaps (FVG) from the numbers.
3. Use Monte Carlo-style reasoning to provide 3 probabilistic scenarios with percentages.
4. Recommend a Trade Plan: Entry, Stop Loss (using ATR), and Take Profit.

### DATA INPUT:
{data_summary}

### OUTPUT FORMAT:
- BIAS: [Result]
- PROBABILITY: [Bullish % | Bearish % | Sideways %]
- NARRATIVE: [Brief logic]
- ACTION: [Buy/Sell/Wait]
- LEVELS: [Entry, SL, TP]
"""

    try:
        res = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1200,
            temperature=0.2, # ต่ำเพื่อให้วิเคราะห์ตัวเลขได้แม่นยำ
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content[0].text
    except Exception as e:
        return f"Mini-Kronos Analysis Failed: {str(e)}"

# --- 5. MAIN EXECUTION ---
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    
    for tf, cfg in TIMEFRAMES.items():
        try:
            df = safe_download(SYMBOL, cfg["interval"], cfg["period"])
            if cfg["resample"]: 
                df = resample_ohlcv(df, cfg["resample"])
            df = add_indicators(df)
            results.append(TimeframeResult(tf, df))
            logger.info(f"Successfully processed {tf}")
        except Exception as e:
            logger.error(f"Failed to process {tf}: {e}")

    if results:
        analysis = call_claude(results)
        print("\n=== MINI-KRONOS GOLD ANALYSIS ===\n")
        print(analysis)
        # บันทึกผลลัพธ์ลงไฟล์เพื่อใช้ใน GitHub Actions
        RESULT_FILE.write_text(analysis, encoding="utf-8")
    else:
        logger.error("No data available for analysis.")

if __name__ == "__main__":
    main()
