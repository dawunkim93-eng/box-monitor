"""별도 테마 — 바닥 추세 전환 매매 검증 (2년)

진입: 5/20일선 골든크로스(최근 5일) + 거래량 1.5배 + 거래대금 10억 이상 + 종가 > 20일선
청산: 목표 +N% / 손절 -M% / 최대 보유 D거래일 (다음날 시가 진입)
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
MIN_AMT = 10.0


def simulate(o, c, h, l, e, tgt, stop, maxd):
    px = float(o[e])
    if px <= 0:
        return None
    for d in range(maxd):
        j = e + d
        if j >= len(c):
            return None
        if h[j] >= px * (1 + tgt):
            return tgt - FEE, d + 1
        if l[j] <= px * (1 - stop):
            return -stop - FEE, d + 1
    j = min(e + maxd - 1, len(c) - 1)
    return float(c[j]) / px - 1 - FEE, maxd


codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
sigs = []
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
    ma5 = cs.rolling(5).mean().values
    ma20 = cs.rolling(20).mean().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    vol5 = pd.Series(v).rolling(5).mean().values
    vol20 = pd.Series(v).rolling(20).mean().values
    for i in range(270, n - 2):
        if df.index[i] < start or amt20[i] < MIN_AMT:
            continue
        if not (ma5[i] > ma20[i] and c[i] > ma20[i]):
            continue
        if not any(ma5[k] > ma20[k] and ma5[k - 1] <= ma20[k - 1]
                   for k in range(i - 4, i + 1)):
            continue
        if vol20[i] <= 0 or vol5[i] / vol20[i] < 1.5:
            continue
        sigs.append((o, c, h, l, i + 1))

print(f"[INFO] 추세 전환 신호 {len(sigs):,}건 · 종목 {len(set(id(s[0]) for s in sigs)):,}")

print(f"\n{'목표/손절/보유':<22}{'건수':>7}{'승률':>7}{'건당%':>8}{'합계%':>9}{'평균보유':>8}")
for tgt, stop, maxd in [(0.03, 0.05, 10), (0.05, 0.05, 10), (0.05, 0.08, 15),
                         (0.08, 0.08, 20), (0.10, 0.10, 20), (0.05, 0.05, 20),
                         (0.03, 0.03, 5), (0.20, 0.10, 40)]:
    out = [simulate(o, c, h, l, e, tgt, stop, maxd) for o, c, h, l, e in sigs]
    out = [x for x in out if x]
    if not out:
        continue
    rets = np.array([x[0] for x in out]) * 100
    days = np.mean([x[1] for x in out])
    print(f"{tgt*100:.0f}% / -{stop*100:.0f}% / {maxd}일{'':<6}"
          f"{len(rets):>7,}{(rets>0).mean()*100:>7.1f}%{rets.mean():>+8.3f}"
          f"{rets.sum():>+9.0f}{days:>8.1f}")
