"""진입 시각 검증 (5분봉 55일) — 표본을 넓혀서

분봉(네이버)은 9일치뿐이라 5분봉 55일치로 다시 본다.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
M5 = ROOT / "data" / "min5"
STOP, TGT = 0.035, 0.020
MIN_AMT, MAX_AMT, POS_MIN = 10.0, 37.51, 89.47
SURGE, SURGE1 = 40.0, 20.0
TIMES = ["0900", "0930", "1000", "1030", "1100", "1200", "1300", "1400", "1500"]


def load5(code):
    p = M5 / f"{code}.csv.gz"
    if not p.exists():
        return {}
    out = {}
    with gzip.open(p, "rt") as f:
        for line in f:
            t = line.strip().split(",")
            if len(t) == 2 and t[0].isdigit() and len(t[0]) == 12:
                out[t[0]] = float(t[1])
    return out


codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
first_day, last_day = "29991231", "00000000"
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
    c = df["Close"].values.astype(float); v = df["Volume"].values.astype(float)
    h = df["High"].values.astype(float); l = df["Low"].values.astype(float)
    cs = pd.Series(c)
    lo30 = cs.rolling(30).min().values; hi30 = cs.rolling(30).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    days = [int(x.strftime("%Y%m%d")) for x in df.index]
    m5 = None
    for i in range(270, len(df) - 4):
        d0 = days[i]
        if not (MIN_AMT <= amt20[i] <= MAX_AMT) or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if pos < POS_MIN:
            continue
        if (c[i] / c[i - 3] - 1) * 100 > SURGE or (c[i] / c[i - 1] - 1) * 100 > SURGE1:
            continue
        buy_day = str(days[i + 1])
        if m5 is None:
            m5 = load5(code)
        db = sorted([(t, p) for t, p in m5.items() if t[:8] == buy_day])
        if len(db) < 30:
            continue
        rows.append(dict(code=code, day=buy_day, db=db,
                         later=[(h[j], l[j], c[j]) for j in range(i + 2, min(i + 5, len(df)))],
                         last_close=c[min(i + 3, len(c) - 1)]))

print(f"[INFO] 5분봉 확인 신호 {len(rows):,}건")

res = {tm: [] for tm in TIMES}
for r in rows:
    db = r["db"]
    for tm in TIMES:
        p0 = None
        for t, p in db:
            if t[8:12] <= tm:
                p0 = p
            else:
                break
        if not p0 or p0 <= 0:
            continue
        stop_px, tgt_px = p0 * (1 - STOP), p0 * (1 + TGT)
        out = None
        for t, p in db:
            if t[8:12] <= tm:
                continue
            if p <= stop_px:
                out = -STOP
                break
            if p >= tgt_px:
                out = TGT
                break
        if out is None:
            for hh, ll, cc in r["later"]:
                if hh >= tgt_px:
                    out = TGT
                    break
                if ll <= stop_px:
                    out = -STOP
                    break
            if out is None:
                out = r["last_close"] / p0 - 1
        res[tm].append(out)

print(f"\n{'진입시각':<10}{'건수':>7}{'승률':>8}{'건당%':>9}{'합계%':>9}{'당일손절':>9}")
for tm in TIMES:
    x = np.array(res[tm]) * 100
    if not len(x):
        continue
    stops = (x <= -STOP * 100 + 1e-9).mean() * 100
    print(f"{tm[:2]}:{tm[2:]:<6}{len(x):>7,}{(x>0).mean()*100:>7.1f}%{x.mean():>+9.3f}"
          f"{x.sum():>+9.0f}{stops:>8.0f}%")

# 시간대별 변동성 (5분봉 기준)
vol = {}
for r in rows[:600]:
    db = r["db"]
    for k in range(1, len(db)):
        t, p = db[k]
        pp = db[k - 1][1]
        if pp > 0:
            vol.setdefault(t[8:10], []).append(abs(p / pp - 1) * 100)
print(f"\n{'시간대':<8}{'5분당 평균 변동':>16}")
for hh in sorted(vol):
    a = np.array(vol[hh])
    print(f"{hh}:00{'':<4}{a.mean():>14.4f}%")
