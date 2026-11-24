"""재료 급등 후 조정 매수 — 조건별 재검증

V1 현행: 조정 -8% 이상 + (20일선 근접 or -15%) + 반등
V2 조정 후 20일선 위 유지 + 20일선 상승 + 반등
V3 조정 폭 -8~-15%로 제한 + 반등
V4 전고점 재돌파 매수 (조정 후 고점 돌파)
V5 거래대금 30억 이상 + V2
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
FEE = 0.00015
MONTHS = 24


def sim(o, h, l, c, e, tgt, stop, maxd):
    px = float(o[e])
    if px <= 0:
        return None
    for d in range(maxd):
        j = e + d
        if j >= len(c):
            return None
        if h[j] >= px * (1 + tgt):
            return tgt - FEE
        if l[j] <= px * (1 - stop):
            return -stop - FEE
    j = min(e + maxd - 1, len(c) - 1)
    return float(c[j]) / px - 1 - FEE


codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
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
    o = df["Open"].values.astype(float)
    c = df["Close"].values.astype(float); h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float); v = df["Volume"].values.astype(float)
    n = len(df)
    cs = pd.Series(c)
    ma20 = cs.rolling(20).mean().values
    v20 = pd.Series(v).rolling(20).mean().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    for i in range(270, n - 20):
        if df.index[i] < start or amt20[i] < 5.0 or v20[i] <= 0:
            continue
        up = (c[i] / c[i - 1] - 1) * 100
        vr = v[i] / v20[i]
        if up < 10 or vr < 5:
            continue
        pk = c[i]
        for j in range(i + 2, min(i + 21, n - 2)):
            if c[j] > pk * 1.05:
                pk = c[j]
            dd = (c[j] / pk - 1) * 100
            near20 = ma20[j] > 0 and abs(c[j] / ma20[j] - 1) <= 0.05
            above20 = ma20[j] > 0 and c[j] > ma20[j]
            rising20 = ma20[j] > ma20[j - 5]
            rebound = c[j] > c[j - 1]
            e = j + 1
            if e >= n:
                break

            v1 = dd <= -8 and (near20 or dd <= -15) and rebound
            v2 = dd <= -5 and above20 and rising20 and rebound
            v3 = -15 <= dd <= -8 and rebound
            v4 = dd <= -5 and c[j] > pk * 0.98        # 전고점 재돌파
            if v1 or v2 or v3 or v4:
                rows.append(dict(
                    v1=v1, v2=v2, v3=v3, v4=v4,
                    amt=amt20[i],
                    a=sim(o, h, l, c, e, 0.10, 0.08, 15),
                    b=sim(o, h, l, c, e, 0.08, 0.08, 10),
                    d=sim(o, h, l, c, e, 0.15, 0.10, 20)))
            if v1 or v2 or v3 or v4:
                break

d = pd.DataFrame(rows)
print(f"[INFO] 후보 {len(d):,}건")


def show(mask, label, col="a"):
    x = d[mask][col].dropna().values * 100
    if not len(x):
        return
    print(f"{label:<34}{len(x):>7,}{(x>0).mean()*100:>7.1f}%{x.mean():>+8.3f}{x.sum():>+8.0f}")


print(f"\n{'조건 (10% / -8% / 15일)':<34}{'건수':>7}{'승률':>7}{'건당%':>8}{'합계%':>8}")
show(d.v1, "V1 조정 후 반등 (현행)")
show(d.v2, "V2 20일선 위 유지 + 20일선 상승")
show(d.v3, "V3 조정 -8~-15% 제한")
show(d.v4, "V4 전고점 재돌파")
show(d.v2 & (d.amt >= 30), "V2 + 거래대금 30억 이상")
show(d.v2 & (d.amt < 30), "V2 + 거래대금 30억 미만")
