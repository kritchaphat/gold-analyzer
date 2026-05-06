import os
import anthropic
import pandas as pd
import yfinance as yf
import requests
from dataclasses import dataclass
from typing import List

# --- SETTINGS ---
SYMBOL = "GC=F" # Gold Futures

@dataclass
class TimeframeResult:
    name: str
    df: pd.DataFrame

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))
    return df.dropna()

def send_telegram(message: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials missing")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    res = requests.post(url, json=payload)
    if res.status_code == 200:
        print("Successfully sent to Telegram!")
    else:
        print(f"Failed to send: {res.text}")

def call_claude(results: List[TimeframeResult]) -> str:
    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
    summary = ""
    for r in results:
        last = r.df.iloc[-1]
        summary += f"*{r.name}* Price: {last['Close']:.2f}, RSI: {last['RSI_14']:.1f}, ATR: {last['ATR']:.2f}\n"
    
    # แก้ model เป็นชื่อที่มีอยู่จริง (3.5 Sonnet คือตัวที่เก่งที่สุดตอนนี้)
    prompt = f"Analyze XAU/USD using SMC (Order Blocks/Liquidity):\n{summary}\nProvide: Bias, Entry, SL (use ATR), TP. Focus on London/NY sessions."
    res = client.messages.create(
        model="claude-3-5-sonnet-20240620", 
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.content[0].text

def main():
    tfs = {"H4": "4h", "H1": "1h", "M30": "30m", "M15": "15m"}
    results = []
    for name, interval in tfs.items():
        df = yf.Ticker(SYMBOL).history(period="10d", interval=interval)
        if not df.empty:
            results.append(TimeframeResult(name, add_indicators(df)))
    
    if results:
        analysis = call_claude(results)
        final_msg = f"🏆 *Gold SMC Analysis Report*\n\n{analysis}"
        send_telegram(final_msg)

if __name__ == "__main__":
    main()
