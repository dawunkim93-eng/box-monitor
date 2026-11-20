"""분봉으로 아침 매수 체결 시점 확인

가정: 시가가 전일 종가보다 높으면 전일 종가에 지정가.
질문: 그 지정가가 장 초반에 체결되는가, 아니면 늦게 체결되는가.
"""
import csv
import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
MDIR = ROOT / "data" / "minute"

files = sorted(MDIR.glob("*.csv.gz"))
rows = []
for f in files:
    code = f.name.replace(".csv.gz", "")
    try:
        pit = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        continue
    pit = pit[pit["Volume"] > 0].sort_index()
    if len(pit) < 2:
        continue
    closes = {int(d.strftime("%Y%m%d")): float(v) for d, v in zip(pit.index, pit["Close"].values)}
    days = sorted(closes)

    bars = {}
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            ts = parts[0]
            d = int(ts[:8])
            bars.setdefault(d, []).append((ts[8:], float(parts[1])))
    for d, arr in bars.items():
        prev = None
        for x in days:
            if x < d:
                prev = closes[x]
            else:
                break
        if prev is None or not arr:
            continue
        arr.sort()
        o = arr[0][1]
        if o < prev:
            rows.append(dict(code=code, day=d, kind="시가매수", wait=0.0))
            continue
        hit = None
        for hhmm, px in arr:
            if px <= prev:
                hit = hhmm
                break
        if hit is None:
            rows.append(dict(code=code, day=d, kind="미체결", wait=None))
        else:
            mins = (int(hit[:2]) - 9) * 60 + int(hit[2:4])
            rows.append(dict(code=code, day=d, kind="지정가체결", wait=max(mins, 0)))

df = pd.DataFrame(rows)
print(f"[INFO] 종목일 {len(df):,}건 · 종목 {df['code'].nunique():,}")
print(df["kind"].value_counts().to_string())

lim = df[df["kind"] == "지정가체결"]
if len(lim):
    for m in (1, 5, 10, 30, 60, 120):
        print(f"  {m:>3}분 내 체결 {100*(lim['wait'] <= m).mean():5.1f}%")
    print(f"  중간값 {lim['wait'].median():.0f}분 · 평균 {lim['wait'].mean():.0f}분")
