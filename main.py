import os
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
    # คำนวณ ATR & RSI เบื้องต้น
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

def main():
    # ดึงข้อมูล 4 ไทม์เฟรมตามระบบของอาร์ม
    tfs = {"H4": "4h", "H1": "1h", "M30": "30m", "M15": "15m"}
    results = []
    
    status_report = "🚀 *Gold Bot System Check*\n\n"
    status_report += "เชื่อมต่อสำเร็จ! ข้อมูลราคาปัจจุบัน:\n"
    
    for name, interval in tfs.items():
        df = yf.Ticker(SYMBOL).history(period="10d", interval=interval)
        if not df.empty:
            data = add_indicators(df)
            last = data.iloc[-1]
            results.append(TimeframeResult(name, data))
            # สร้างรายงานราคาแบบไม่ต้องง้อ AI
            status_report += f"📍 *{name}*: {last['Close']:.2f} (RSI: {last['RSI_14']:.1f})\n"
    
    status_report += "\n✅ *ระบบพร้อมแล้ว!* ถ้าอาร์มเห็นข้อความนี้ แปลว่า Telegram เชื่อมต่อกับ GitHub Actions สมบูรณ์ 100% โดยไม่เสีย Token ครับ"
    
    if results:
        print("Sending system status to Telegram...")
        send_telegram(status_report)

if __name__ == "__main__":
    main()
