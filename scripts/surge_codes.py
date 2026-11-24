"""재료일 신호 종목 추출 → DART 공시 수집까지 한 번에"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
OUT = ROOT / "data" / "surge_signals.json"

codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
rows = []
for code in codes:
    if code[-1] in "579K":
        continue
    try:
        df = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        continue
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Volume"] > 0]
    if len(df) < 320:
        continue
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float)
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    v20 = pd.Series(v).rolling(20).mean().values
    for i in range(270, len(df) - 1):
        if amt20[i] < 5 or v20[i] <= 0 or c[i - 1] <= 0:
            continue
        if (c[i] / c[i - 1] - 1) * 100 >= 10 and v[i] / v20[i] >= 5:
            rows.append(dict(code=code, day=int(df.index[i].strftime("%Y%m%d"))))

OUT.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
uniq = sorted({r["code"] for r in rows})
print(f"[INFO] 재료일 신호 {len(rows):,}건 · 종목 {len(uniq)}")
(ROOT / "data" / "surge_codes.json").write_text(json.dumps(uniq), encoding="utf-8")
