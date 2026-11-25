"""지정가를 전일 종가에서 얼마나 아래에 걸 것인가

A) 현행: 시가 < 전일종가 → 시가 매수, 아니면 전일종가 지정가
B) 전일종가 -0.5% / -1.0% / -1.5% / -2.0% 지정가 (시가가 그보다 낮으면 시가 매수)
C) 전일종가 -1.5% 지정가만 (시가가 낮아도 지정가 대기)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
BUY_GAP, SELL_GAP, FEE = 0.015, 0.020, 0.00015
AUTO_STOP = 0.035
SLIP_BUY = 0.0005      # 매수 체결 0.05% 불리
SLIP_STOP = 0.003      # 자동손절 0.3% 불리
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_AMT, MAX_AMT, POS_MIN = 10.0, 37.51, 89.47
SURGE, SURGE1 = 40.0, 20.0
MONTHS = 24
VARIANTS = [("A 현행", None), ("B -0.5%", 0.005), ("B -1.0%", 0.010),
            ("B -1.5%", 0.015), ("B -2.0%", 0.020)]


def trade(o, c, h, l, i, disc, limit_only=False):
    """disc: 전일종가 대비 지정가 하향(비율). None이면 현행."""
    e = i + 1
    if e >= len(c):
        return None
    prev = float(c[i])
    if disc is None:
        if float(o[e]) < prev:
            px = float(o[e])
        elif float(l[e]) <= prev:
            px = prev
        else:
            return None
    else:
        lv_px = prev * (1 - disc)
        if (not limit_only) and float(o[e]) < lv_px:
            px = float(o[e])            # 시가가 더 낮으면 시가 체결
        elif float(l[e]) <= lv_px:
            px = lv_px
        else:
            return None
    px *= (1 + SLIP_BUY)
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    stop_px = px * (1 - AUTO_STOP)
    limit_entry = True                  # 지정가 성격(첫날 시가 매도 판정 보수적으로)
    if disc is None and abs(px - float(o[e]) * (1 + SLIP_BUY)) < 1e-6:
        limit_entry = False
    held = {0: px}
    net = 0.0
    for dd in range(HOLD):
        j = e + dd
        if j >= len(c):
            break
        for k in (1, 2):
            if k not in held and float(l[j]) <= lv[k]:
                held[k] = lv[k]
        for k in list(held):
            hit = (float(h[j]) >= held[k] * (1 + SELL_GAP)) if (dd > 0 or not limit_entry) \
                else (float(c[j]) >= held[k] * (1 + SELL_GAP))
            if hit:
                net += AMTS[k] * 10_000 * (SELL_GAP - FEE)
                del held[k]
        if held and float(l[j]) <= stop_px:
            fill = min(stop_px, float(o[j])) if float(o[j]) < stop_px else stop_px
            for k in list(held):
                net += AMTS[k] * 10_000 * (fill / held[k] - 1 - FEE - SLIP_STOP)
            return net
        if held and float(c[j]) < lv[2]:
            for k in list(held):
                net += AMTS[k] * 10_000 * (float(c[j]) / held[k] - 1 - FEE)
            return net
    if held:
        j = min(e + HOLD - 1, len(c) - 1)
        for k in list(held):
            net += AMTS[k] * 10_000 * (float(c[j]) / held[k] - 1 - FEE)
    return net


codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
res = {name: [] for name, _ in VARIANTS}
sigs = 0
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
    cs = pd.Series(c)
    lo30 = cs.rolling(30).min().values; hi30 = cs.rolling(30).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    for i in range(270, len(df) - 1):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        if not (MIN_AMT <= amt20[i] <= MAX_AMT):
            continue
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if pos < POS_MIN:
            continue
        if (c[i] / c[i - 3] - 1) * 100 > SURGE or (c[i] / c[i - 1] - 1) * 100 > SURGE1:
            continue
        sigs += 1
        for name, disc in VARIANTS:
            r = trade(o, c, h, l, i, disc)
            if r is not None:
                res[name].append(r)

print(f"[INFO] 신호 {sigs:,}건\n")
print(f"{'방식':<12}{'체결':>7}{'체결률':>8}{'승률':>8}{'건당%':>9}{'합계(원)':>13}")
for name, _ in VARIANTS:
    x = np.array(res[name])
    if not len(x):
        continue
    pct = x / SLOT * 100
    print(f"{name:<12}{len(x):>7,}{len(x)/sigs*100:>7.1f}%{(x>0).mean()*100:>7.1f}%"
          f"{pct.mean():>+9.3f}{x.sum():>+13,.0f}")
