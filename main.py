import os
import anthropic
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from pathlib import Path
from typing import List

# --- SETTINGS ---
SYMBOL = "GC=F" # Gold Futures (XAU/USD)
OUTPUT_DIR = Path("output")
RESULT_FILE = OUTPUT_DIR / "analysis_result.txt"

@dataclass
class TimeframeResult:
    name: str
    df: pd.DataFrame

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    # EMA
    df['EMA_9']  = df['Close'].ewm(span=9,  adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))
    return df.dropna()

def call_claude(results: List[TimeframeResult]) -> str:
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        return "Error: No API Key found"
    
    client = anthropic.Anthropic(api_key=api_key)
    
    summary = ""
    for r in results:
        last = r.df.iloc[-1]
        summary += f"[{r.name}] Price: {last['Close']:.2f}, RSI: {last['RSI_14']:.1f}, ATR: {last['ATR']:.2f}\n"
    
    prompt = f"Analyze XAU/USD using SMC (Order Blocks/Liquidity):\n{summary}\nProvide: Bias, Entry, SL, TP."

    # ใช้ชื่อโมเดลล่าสุดจากหน้า Docs ของอาร์มปี 2026
    res = client.messages.create(
        model="claude-opus-4-7", 
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.content[0].text

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    tfs = {"H4": "4h", "H1": "1h", "M30": "30m", "M15": "15m"}
    results = []
    
    for name, interval in tfs.items():
        print(f"Fetching {name}...")
        ticker = yf.Ticker(SYMBOL)
        df = ticker.history(period="10d", interval=interval)
        if df.empty: continue
        if name == "H4":
            df = df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        results.append(TimeframeResult(name, add_indicators(df)))
    
    if not results:
        print("Data fetch error"); return

    print("Analyzing with Claude Opus 4.7...")
    analysis = call_claude(results)
    print(f"\n{analysis}")
    RESULT_FILE.write_text(analysis, encoding="utf-8")

if __name__ == "__main__":
    main()
