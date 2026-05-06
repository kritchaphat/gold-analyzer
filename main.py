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
    
    # 1. ATR (หา SL/TP)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # 2. EMA 3 เส้น (Trend Analysis)
    df['EMA_9']  = df['Close'].ewm(span=9,  adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # 3. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))
    
    # 4. MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 5. Bollinger Bands
    df['BB_Mid']   = df['Close'].rolling(20).mean()
    df['BB_Std']   = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    
    return df.dropna()

def call_claude(results: List[TimeframeResult]) -> str:
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        return "Error: No API Key found in Environment"
    
    client = anthropic.Anthropic(api_key=api_key)
    
    summary = ""
    for r in results:
        last = r.df.iloc[-1]
        ema_trend = "BULLISH" if last['EMA_9'] > last['EMA_21'] > last['EMA_50'] \
                    else "BEARISH" if last['EMA_9'] < last['EMA_21'] < last['EMA_50'] \
                    else "RANGING"
        bb_pos = "UPPER" if last['Close'] >= last['BB_Upper'] * 0.999 \
                 else "LOWER" if last['Close'] <= last['BB_Lower'] * 1.001 \
                 else "MID"
        macd_dir = "BULLISH" if last['MACD_Hist'] > 0 else "BEARISH"
        
        summary += f"""
[{r.name}]
Price  : {last['Close']:.2f}
EMA    : 9={last['EMA_9']:.2f} | 21={last['EMA_21']:.2f} | 50={last['EMA_50']:.2f}
RSI    : {last['RSI_14']:.1f}
MACD   : Hist={last['MACD_Hist']:.3f} ({macd_dir})
BB     : Position={bb_pos}
ATR    : {last['ATR']:.2f}
Trend  : {ema_trend}
"""
    
    prompt = f"""You are a Senior XAU/USD Quantitative Strategist specializing in SMC, Order Blocks, and Liquidity Analysis.
Analyze the following multi-timeframe data and provide a trade plan:
{summary}
Please provide:
1. HTF Bias (H4 trend direction and reasoning)
2. Liquidity Levels (potential BSL/SSL targets)
3. Key Supply/Demand Zones (Order Blocks)
4. Trade Setup: Direction, Entry, SL (ATR x 1.5), TP1, TP2, TP3
5. Invalidation condition
Keep response concise and professional."""

    # อัปเดต Model Name ตามที่อาร์มขอจากรูปภาพ
    res = client.messages.create(
        model="claude-sonnet-4-20250514",
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
            df = df.resample('4h').agg({
                'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'
            }).dropna()
        df = add_indicators(df)
        results.append(TimeframeResult(name, df))
    
    if not results:
        print("Error: No data retrieved.")
        return

    print(f"Analyzing with Claude Sonnet 4...")
    analysis = call_claude(results)
    print("\n" + "="*50 + "\n" + analysis + "\n" + "="*50)
    RESULT_FILE.write_text(analysis, encoding="utf-8")

if __name__ == "__main__":
    main()
