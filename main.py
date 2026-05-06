import os
import anthropic
import pandas as pd
import yfinance as yf
import requests
from dataclasses import dataclass
from typing import List

# --- CONFIGURATION ---
SYMBOL = "GC=F"  # Gold Futures (XAU/USD)

@dataclass
class TimeframeResult:
    name: str
    df: pd.DataFrame

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """คำนวณ ATR สำหรับตั้ง SL และ RSI สำหรับเช็ค Momentum"""
    df = df.copy()
    # ATR Calculation
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))
    return df.dropna()

def send_telegram(message: str):
    """ส่งข้อความเข้า Telegram Bot"""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Error: Telegram credentials missing")
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
    """เรียกใช้ Claude Opus 4.7 เพื่อวิเคราะห์หน้าเทรดตามระบบ SMC"""
    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
    
    # เตรียมข้อมูลส่งให้ AI
    market_data = ""
    for r in results:
        last = r.df.iloc[-1]
        market_data += f"- {r.name}: Price {last['Close']:.2f}, RSI {last['RSI_14']:.1f}, ATR {last['ATR']:.2f}\n"
    
    # Prompt ที่อาร์มเคยใช้ทดสอบระบบ SMC
    prompt = f"""คุณคือผู้เชี่ยวชาญการเทรด XAU/USD ด้วยระบบ Smart Money Concepts (SMC) 
วิเคราะห์ข้อมูลล่าสุดนี้:
{market_data}

กรุณาให้ข้อมูลตามโครงสร้างนี้:
1. Market Structure (H4/H1/M15)
2. Bias (Bullish/Bearish/Neutral)
3. Trade Setup: ระบุ Entry Zone (Order Block), Stop Loss (อิงตาม ATR), และ Take Profit (3 ระดับ)
4. Key Confirmation: เงื่อนไขที่ต้องรอก่อนเข้าเทรด (เช่น BOS หรือ Liquidity Sweep)

แสดงผลด้วย Markdown ที่สวยงาม"""

    res = client.messages.create(
        model="claude-3-7-sonnet-20250219", # หรือใช้ opus ตามที่ตั้งค่าไว้
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.content[0].text

def main():
    print("Fetching market data...")
    tfs = {"H4": "4h", "H1": "1h", "M30": "30m", "M15": "15m"}
    results = []
    
    for name, interval in tfs.items():
        # ดึงข้อมูลย้อนหลัง 10 วันเพื่อให้ Indicator คำนวณได้แม่นยำ
        df = yf.Ticker(SYMBOL).history(period="10d", interval=interval)
        if not df.empty:
            results.append(TimeframeResult(name, add_indicators(df)))
    
    if results:
        print("Analyzing with Claude...")
        analysis = call_claude(results)
        
        final_msg = f"🏆 *Gold SMC Analysis Report*\n\n{analysis}"
        send_telegram(final_msg)

if __name__ == "__main__":
    main()
