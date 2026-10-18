"""Download the public BBC 2004-2005 news dataset into the app's data folder.

Usage:
    python download_week7_news.py
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import requests
import pandas as pd

SOURCE = (
    "https://raw.githubusercontent.com/suraj-deshmukh/"
    "BBC-Dataset-News-Classification/master/dataset/dataset.csv"
)
OUTPUT = Path(__file__).resolve().parent / "data" / "week7_bbc_news.csv"


def main() -> None:
    print("Downloading historical BBC news dataset...")
    try:
        response = requests.get(SOURCE, timeout=60)
        response.raise_for_status()
        data = response.content
        df = pd.read_csv(BytesIO(data), encoding_errors="replace")
        if not {"news", "type"}.issubset({col.strip().lower() for col in df.columns}):
            raise ValueError("Downloaded file does not contain expected news/type columns")
        if len(df) < 100:
            raise ValueError("Downloaded dataset appears incomplete")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(data)
        print(f"Saved {len(df):,} rows to {OUTPUT}")
    except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(
            f"Dataset download failed: {exc}\n"
            f"Download the CSV manually from:\n{SOURCE}\n"
            f"Place it at: {OUTPUT}"
        ) from exc


if __name__ == "__main__":
    main()
