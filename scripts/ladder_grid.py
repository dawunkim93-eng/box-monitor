"""사다리 간격 비교 — 매수 간격 × 매도 간격 (2년, 아침 매수 기준)

1차는 매수일에 체결, 2·3차는 지정가(-간격, -2×간격).
각 차수 매도는 매수가 × (1+매도간격).
종가가 3차 아래면 그날 청산, 매수일부터 3거래일 만기.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
FEE = 0.00015
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_AMT, MAX_AMT, POS_MIN, SURGE = 10.0, 37.51, 89.47, 20.0
MONTHS = 24

BUY_GAPS = [1.5, 2.0, 2.5, 3.0]
SELL_GAPS = [1.5, 1.7, 2.0, 2.5]


def trade(o, c, h, l, e, px, bg, sg):
    """e: 매수일 인덱스, px: 1차 체결가"""
    lv = [px * (1 - bg / 100) ** k for k in range(3)]
    tg = [x * (1 + sg / 100) for x in lv]
    held = {0: px}
    net = 0.0
    for dd in range(HOLD):
        j = e + dd
        if j >= len(c):
            break
        for k in (1, 2):
            if k not in held and l[j] <= lv[k]:
                held[k] = lv[k]
        for k in list(held):
            if h[j] >= tg[k]:
                net += AMTS[k] * 10_000 * (sg / 100 - FEE)
                del held[k]
        if held and c[j] < lv[2]:
            for k in list(held):
                net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
                del held[k]
            break
    if held:
        j = min(e + HOLD - 1, len(c) - 1)
        for k in list(held):
            net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
    return net / SLOT * 100


codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
packs = []
sig_idx = []

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
    lo30 = cs.rolling(30).min().values; hi30 = cs.rolling(30).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    pid = len(packs)
    packs.append((o, c, h, l))
    for i in range(270, n - 2):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        a = amt20[i]
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if not (MIN_AMT <= a <= MAX_AMT) or pos < POS_MIN:
            continue
        if (c[i] / c[i - 3] - 1) * 100 > SURGE:
            continue
        e = i + 1
        prev = float(c[i])
        if float(o[e]) < prev:
            px = float(o[e])
        elif float(l[e]) <= prev:
            px = prev
        else:
            continue                        # 미체결
        sig_idx.append((pid, e, px))

print(f"[INFO] 체결 신호 {len(sig_idx):,}건 · 종목 {len(packs):,}")
print(f"\n{'매수':>6}{'매도':>6}{'건수':>8}{'승률':>7}{'건당%':>9}{'합계%':>9}{'-5%↓':>7}")
best = []
for bg in BUY_GAPS:
    for sg in SELL_GAPS:
        vals = np.array([trade(*packs[pid], e, px, bg, sg) for pid, e, px in sig_idx])
        print(f"{bg:>5.1f}%{sg:>5.1f}%{len(vals):>8,}{(vals>0).mean()*100:>7.1f}%"
              f"{vals.mean():>+9.3f}{vals.sum():>+9.0f}{(vals<=-5).mean()*100:>6.1f}%")
        best.append((vals.mean(), bg, sg, vals.sum(), (vals > 0).mean() * 100))
    print()
best.sort(reverse=True)
print("[상위]")
for m, bg, sg, tot, win in best[:5]:
    print(f"  매수 {bg}% / 매도 {sg}% · 건당 {m:+.3f}% · 승률 {win:.1f}% · 합계 {tot:+.0f}%")
