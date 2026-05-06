import os
import anthropic
import pandas as pd
import yfinance as yf
import requests
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
    """คำนวณ Indicators: ATR, EMA 9/21/50 และ RSI"""
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

def send_telegram(message: str):
    """ส่งข้อความเข้า Telegram"""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ Telegram credentials missing - skipping notification")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload)
        if res.status_code == 200:
            print("✅ Successfully sent to Telegram!")
        else:
            print(f"❌ Failed to send Telegram: {res.text}")
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

def call_claude(results: List[TimeframeResult]) -> str:
    """วิเคราะห์ด้วย Claude Opus 4.7"""
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        return "Error: No API Key found"
    
    client = anthropic.Anthropic(api_key=api_key)
    
    summary = ""
    for r in results:
        last = r.df.iloc[-1]
        summary += (f"[{r.name}] Price: {last['Close']:.2f}, RSI: {last['RSI_14']:.1f}, "
                   f"EMA9/21/50: {last['EMA_9']:.2f}/{last['EMA_21']:.2f}/{last['EMA_50']:.2f}, "
                   f"ATR: {last['ATR']:.2f}\n")
    
    # Prompt เน้น SMC และวินัยตามช่วงเวลา
    prompt = (f"คุณคือผู้เชี่ยวชาญ SMC วิเคราะห์ XAU/USD จากข้อมูลนี้:\n{summary}\n"
              f"กรุณาระบุ: Bias, Entry Zone (Order Block), SL, TP 3 ระดับ\n"
              f"เน้นย้ำวินัย: รอ Confirmation ใน M5/M1 และเทรดเฉพาะช่วง London/NY Session เท่านั้น")

    res = client.messages.create(
        model="claude-opus-4-7", # ใช้ชื่อโมเดลตามที่อาร์มต้องการ
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.content[0].text

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    tfs = {"H4": "4h", "H1": "1h", "M30": "30m", "M15": "15m", "M5": "5m"}
    results = []
    
    for name, interval in tfs.items():
        print(f"Fetching {name}...")
        ticker = yf.Ticker(SYMBOL)
        df = ticker.history(period="10d", interval=interval)
        if df.empty: continue
        
        # จัดการข้อมูล H4 ให้ตรงตาม format
        if name == "H4":
            df = df.resample('4h').agg({
                'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'
            }).dropna()
            
        results.append(TimeframeResult(name, add_indicators(df)))
    
    if not results:
        print("❌ Data fetch error"); return

    print("Analyzing with Claude Opus 4.7...")
    analysis = call_claude(results)
    
    # บันทึกลงไฟล์
    RESULT_FILE.write_text(analysis, encoding="utf-8")
    
    # ส่งเข้า Telegram
    final_msg = f"🏆 *Gold SMC Analysis (Opus 4.7)*\n\n{analysis}"
    send_telegram(final_msg)
    
    print(f"\n--- Analysis Finished ---\n{analysis}")

if __name__ == "__main__":
    main()
