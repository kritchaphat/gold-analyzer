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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("gold-analyzer")

SYMBOL = "XAUUSD=X"
OUTPUT_DIR = Path("output")
CHART_DIR = OUTPUT_DIR / "charts"
RESULT_FILE = OUTPUT_DIR / "analysis_result.txt"
INDICATOR_FILE = OUTPUT_DIR / "indicator_snapshot.json"

TIMEFRAMES = {
    "1H": {"interval": "60m", "period": "730d", "resample": None},
    "30M": {"interval": "30m", "period": "60d", "resample": None},
    "15M": {"interval": "15m", "period": "60d", "resample": None},
    "5M": {"interval": "5m", "period": "60d", "resample": None},
    # 4H is derived from yfinance 1H candles to satisfy intraday constraints.
    "4H": {"interval": "60m", "period": "730d", "resample": "4H"},
}


@dataclass
class TimeframeResult:
    name: str
    df: pd.DataFrame
    chart_path: Path
    latest_indicators: Dict[str, float]


def safe_download(symbol: str, interval: str, period: str, retries: int = 3, sleep_seconds: int = 3) -> pd.DataFrame:
    """Download data from yfinance with retry and validation."""
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s data (interval=%s, period=%s), attempt %s/%s", symbol, interval, period, attempt, retries)
            df = yf.download(
                tickers=symbol,
                interval=interval,
                period=period,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:
                raise ValueError(f"No data returned for {symbol} interval={interval} period={period}")

            # yfinance can return multi-index columns depending on version/options.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            needed = {"Open", "High", "Low", "Close", "Volume"}
            missing = needed.difference(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {sorted(missing)}")

            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) < 60:
                raise ValueError(f"Insufficient candles ({len(df)}) for interval={interval}")

            return df
        except Exception as exc:  # broad by design for robust retries in CI
            last_error = exc
            logger.warning("Download failed: %s", exc)
            if attempt < retries:
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to download data after {retries} attempts: {last_error}")


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV dataframe into a larger timeframe."""
    logger.info("Resampling data to %s", rule)

    out = (
        df.resample(rule)
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna()
    )

    if len(out) < 60:
        raise ValueError(f"Insufficient candles ({len(out)}) after resampling to {rule}")

    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # EMAs
    out["EMA_9"] = out["Close"].ewm(span=9, adjust=False).mean()
    out["EMA_21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()

    # Bollinger Bands (20, 2)
    basis = out["Close"].rolling(window=20).mean()
    dev = out["Close"].rolling(window=20).std(ddof=0)
    out["BB_MID"] = basis
    out["BB_UPPER"] = basis + (2 * dev)
    out["BB_LOWER"] = basis - (2 * dev)

    # RSI (14)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    return out.dropna()


def latest_values(df: pd.DataFrame) -> Dict[str, float]:
    row = df.iloc[-1]
    return {
        "close": float(row["Close"]),
        "ema_9": float(row["EMA_9"]),
        "ema_21": float(row["EMA_21"]),
        "ema_50": float(row["EMA_50"]),
        "bb_upper": float(row["BB_UPPER"]),
        "bb_mid": float(row["BB_MID"]),
        "bb_lower": float(row["BB_LOWER"]),
        "rsi_14": float(row["RSI_14"]),
        "macd": float(row["MACD"]),
        "macd_signal": float(row["MACD_SIGNAL"]),
        "macd_hist": float(row["MACD_HIST"]),
    }


def make_chart(df: pd.DataFrame, timeframe: str, out_path: Path) -> None:
    style = mpf.make_mpf_style(
        base_mpf_style="yahoo",
        gridstyle="--",
        facecolor="#0e1117",
        figcolor="#0e1117",
        edgecolor="#1f2937",
        y_on_right=False,
        rc={"axes.labelcolor": "#d1d5db", "xtick.color": "#d1d5db", "ytick.color": "#d1d5db"},
    )

    plots = [
        mpf.make_addplot(df["EMA_9"], panel=0, color="#22c55e", width=1.0),
        mpf.make_addplot(df["EMA_21"], panel=0, color="#3b82f6", width=1.0),
        mpf.make_addplot(df["EMA_50"], panel=0, color="#f59e0b", width=1.0),
        mpf.make_addplot(df["BB_UPPER"], panel=0, color="#a78bfa", width=0.8),
        mpf.make_addplot(df["BB_MID"], panel=0, color="#e5e7eb", width=0.8),
        mpf.make_addplot(df["BB_LOWER"], panel=0, color="#a78bfa", width=0.8),
        mpf.make_addplot(df["RSI_14"], panel=1, color="#06b6d4", ylabel="RSI(14)", ylim=(0, 100)),
        mpf.make_addplot(pd.Series(70, index=df.index), panel=1, color="#ef4444", width=0.8),
        mpf.make_addplot(pd.Series(30, index=df.index), panel=1, color="#22c55e", width=0.8),
        mpf.make_addplot(df["MACD"], panel=2, color="#60a5fa", ylabel="MACD"),
        mpf.make_addplot(df["MACD_SIGNAL"], panel=2, color="#f97316"),
        mpf.make_addplot(df["MACD_HIST"], panel=2, type="bar", color="#9ca3af", alpha=0.7),
    ]

    mpf.plot(
        df.tail(220),
        type="candle",
        style=style,
        addplot=plots,
        title=f"XAU/USD - {timeframe}",
        volume=False,
        panel_ratios=(6, 2, 2),
        figsize=(14, 10),
        tight_layout=True,
        savefig=dict(fname=str(out_path), dpi=150),
    )


def format_indicator_payload(results: List[TimeframeResult]) -> str:
    payload = {}
    for r in results:
        payload[r.name] = {
            "timestamp_utc": str(r.df.index[-1]),
            **{k: round(v, 5) for k, v in r.latest_indicators.items()},
        }
    return json.dumps(payload, indent=2)


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def call_claude(results: List[TimeframeResult]) -> str:
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise EnvironmentError("CLAUDE_API_KEY is not set in environment variables.")

    client = anthropic.Anthropic(api_key=api_key)

    indicator_json = format_indicator_payload(results)

    content = [
        {
            "type": "text",
            "text": (
                "Act as a professional Gold Trader. Analyze these 5 timeframes to provide a trade setup "
                "for today's London and New York sessions. Identify Trend, Support/Resistance, and "
                "Entry/Exit zones.\n\n"
                "Symbol: XAU/USD\n"
                "Timeframes: 4H, 1H, 30M, 15M, 5M\n"
                "Latest indicator snapshot (JSON):\n"
                f"{indicator_json}"
            ),
        }
    ]

    for item in results:
        content.extend(
            [
                {"type": "text", "text": f"Chart: {item.name}"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_to_base64(item.chart_path),
                    },
                },
            ]
        )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1400,
        temperature=0.2,
        messages=[{"role": "user", "content": content}],
    )

    chunks = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            chunks.append(block.text)

    return "\n".join(chunks).strip()


def process_timeframe(name: str, config: Dict[str, Optional[str]]) -> TimeframeResult:
    raw = safe_download(SYMBOL, interval=config["interval"], period=config["period"])
    if config.get("resample"):
        raw = resample_ohlcv(raw, config["resample"])

    enriched = add_indicators(raw)
    if enriched.empty:
        raise ValueError(f"No enriched data remaining for {name} after indicators.")

    chart_path = CHART_DIR / f"xauusd_{name.lower()}.png"
    make_chart(enriched, name, chart_path)

    return TimeframeResult(
        name=name,
        df=enriched,
        chart_path=chart_path,
        latest_indicators=latest_values(enriched),
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    results: List[TimeframeResult] = []
    errors: List[Tuple[str, str]] = []

    order = ["4H", "1H", "30M", "15M", "5M"]
    for timeframe in order:
        cfg = TIMEFRAMES[timeframe]
        try:
            logger.info("Processing timeframe %s", timeframe)
            result = process_timeframe(timeframe, cfg)
            results.append(result)
            logger.info("Completed timeframe %s -> %s", timeframe, result.chart_path)
        except Exception as exc:  # continue collecting other timeframes
            logger.exception("Failed timeframe %s", timeframe)
            errors.append((timeframe, str(exc)))

    if errors:
        logger.error("Errors encountered in %s timeframe(s): %s", len(errors), errors)

    if len(results) != 5:
        logger.error("Expected 5 successful timeframes, got %s. Aborting AI analysis.", len(results))
        return 1

    indicator_json = format_indicator_payload(results)
    INDICATOR_FILE.write_text(indicator_json, encoding="utf-8")
    logger.info("Saved indicator snapshot: %s", INDICATOR_FILE)

    try:
        analysis = call_claude(results)
        RESULT_FILE.write_text(analysis, encoding="utf-8")
        logger.info("Claude analysis saved to %s", RESULT_FILE)
    except Exception:
        logger.exception("Failed calling Claude API")
        return 1

    print("\n=== Claude Gold Analysis ===\n")
    print(analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
