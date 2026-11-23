"""KOSPI 일봉 동기화 — data/ks11.csv, data/ks11_ohlc.csv"""
import re
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "Mozilla/5.0"}


def main():
    end = pd.Timestamp.now()
    st = (end - pd.Timedelta(days=25)).strftime("%Y%m%d")
    url = ("https://api.finance.naver.com/siseJson.naver?symbol=KOSPI&requestType=1"
           f"&startTime={st}&endTime={end:%Y%m%d}&timeframe=day")
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) \
        .read().decode("utf-8", "ignore")
    rows = re.findall(r'\["(\d{8})",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', raw)
    if not rows:
        print("[INFO] 신규 데이터 없음")
        return
    new = pd.DataFrame([(pd.Timestamp(r[0]), float(r[4])) for r in rows],
                       columns=["Date", "Close"])
    old = pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"])
    merged = pd.concat([old[~old["Date"].isin(new["Date"])], new]).sort_values("Date")
    merged.to_csv(ROOT / "data" / "ks11.csv", index=False)

    ohlc = pd.DataFrame([(pd.Timestamp(r[0]), float(r[1]), float(r[2]),
                          float(r[3]), float(r[4])) for r in rows],
                        columns=["Date", "Open", "High", "Low", "Close"])
    o = pd.read_csv(ROOT / "data" / "ks11_ohlc.csv", parse_dates=["Date"])
    m2 = pd.concat([o[~o["Date"].isin(ohlc["Date"])], ohlc]).sort_values("Date")
    m2.to_csv(ROOT / "data" / "ks11_ohlc.csv", index=False)
    print(f"[INFO] KOSPI {len(merged)}행 · 마지막 {merged['Date'].iloc[-1].date()}")


if __name__ == "__main__":
    main()
