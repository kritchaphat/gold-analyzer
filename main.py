import os
import time
import anthropic
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from pathlib import Path
from typing import List

# --- SETTINGS ---
SYMBOL = "GC=F"
OUTPUT_DIR = Path("output")
RESULT_FILE = OUTPUT_DIR / "analysis_result.txt"

@dataclass
class TimeframeResult:
    name: str
    df: pd.DataFrame

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. คำนวณ ATR แบบ Manual (สูตรมาตรฐาน)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # 2. คำนวณ EMA แบบ Manual
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # 3. คำนวณ RSI แบบ Manual
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    return df.dropna()

def call_claude(results: List[TimeframeResult]) -> str:
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key: return "Error: No API Key found in Environment"
    
    # ใช้ Model ตัวล่าสุดที่ระบบรองรับชัวร์ๆ
    client = anthropic.Anthropic(api_key=api_key)
    
    summary = ""
    for r in results:
        last = r.df.iloc[-1]
        seq = r.df['Close'].tail(10).round(2).tolist()
        summary += f"\n[{r.name}]: Seq={seq}\nLast: C={last['Close']:.2f}, ATR={last['ATR']:.2f}, RSI={last['RSI_14']:.1f}\n"

    prompt = f"Act as Mini-Kronos Gold Expert. Analyze these sequences & indicators: {summary}\nProvide: 1. Bias 2. 3-Scenarios with % 3. Trade Plan (Entry/SL/TP)."
    
    # แก้ไขชื่อ Model เป็นรุ่นล่าสุด
    res = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.content[0].text

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    # ปรับเป็นตัวเล็ก h ตามคำแนะนำของระบบใหม่
    tfs = {"4H": "4h", "1H": "1h", "15M": "15m"}
    results = []
    
    for name, interval in tfs.items():
        print(f"Fetching data for {name}...")
        df = yf.Ticker(SYMBOL).history(period="60d", interval="60m" if "H" in name else "15m")
        
        # ปรับ Resample เป็นตัวเล็ก 4h
        if name == "4H": 
            df = df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
            
        df = add_indicators(df)
        results.append(TimeframeResult(name, df))
    
    print("Calling Claude for Analysis...")
    analysis = call_claude(results)
    print("\n--- ANALYSIS RESULT ---")
    print(analysis)
    
    RESULT_FILE.write_text(analysis, encoding="utf-8")
    print(f"\nSaved to {RESULT_FILE}")

if __name__ == "__main__":
    main()
