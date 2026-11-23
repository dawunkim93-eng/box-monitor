"""자동매도(장중 트리거) 방식 비교 (2년, 아침 매수 · +2.0% 익절)

V0 종가 판정 (현행, 자동매도 없음)
V1 장중 자동매도 1차 -3.5% (3차 체결 가능)
V2 장중 자동매도 1차 -4.0%
V3 3차 체결 후에만 자동매도 1차 -4.5%
V4 3차 체결 후에만 자동매도 1차 -5.0%
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
FEE = 0.00015
BUY_GAP, SELL_GAP = 0.015, 0.020
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_AMT, MAX_AMT, POS_MIN, SURGE = 10.0, 37.51, 89.47, 20.0
MONTHS = 24


def trade(o, c, h, l, e, px, mode, x):
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    tg = [x2 * (1 + SELL_GAP) for x2 in lv]
    held, net, rungs = {0: 0}, 0.0, 1
    armed = mode in ("intraday",)          # intraday: 처음부터 자동매도
    stop_px = px * (1 - x)
    for dd in range(HOLD):
        j = e + dd
        if j >= len(c):
            break
        # 1) 차수 체결 (장중 저가 기준)
        for k in (1, 2):
            if k not in held and l[j] <= lv[k]:
                held[k] = k
                rungs = max(rungs, k + 1)
        if mode == "after3" and rungs == 3:
            armed = True
        # 2) 익절 (장중 고가 기준)
        for k in list(held):
            if h[j] >= tg[k]:
                net += AMTS[k] * 10_000 * (SELL_GAP - FEE)
                del held[k]
        if not held:
            return net, rungs, "익절"
        # 3) 자동매도
        if armed and l[j] <= stop_px:
            fill = min(stop_px, float(o[j])) if float(o[j]) < stop_px else stop_px
            for k in list(held):
                net += AMTS[k] * 10_000 * (fill / lv[k] - 1 - FEE)
            return net, rungs, "자동손절"
        # 4) 종가 판정
        if c[j] < lv[2]:
            for k in list(held):
                net += AMTS[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
            return net, rungs, "종가손절"
    j = min(e + HOLD - 1, len(c) - 1)
    for k in list(held):
        net += AMTS[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
    return net, rungs, "만기"


codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
packs, sigs = [], []
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
    pid = len(packs); packs.append((o, c, h, l))
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
            continue
        sigs.append((pid, e, px))

print(f"[INFO] 체결 신호 {len(sigs):,}건")
cases = [("V0 종가 판정(현행)", "none", 0.03),
         ("V1 장중 -3.5%", "intraday", 0.035),
         ("V2 장중 -4.0%", "intraday", 0.040),
         ("V3 3차후 -4.5%", "after3", 0.045),
         ("V4 3차후 -5.0%", "after3", 0.050),
         ("V5 장중 -3.0%", "intraday", 0.030)]
print(f"\n{'방식':<20}{'승률':>7}{'건당%':>8}{'합계%':>9}{'-5%↓':>7}{'3차체결':>8}{'최악%':>8}")
for lbl, mode, x in cases:
    out = [trade(*packs[pid], e, px, mode, x) for pid, e, px in sigs]
    vals = np.array([z[0] for z in out]) / SLOT * 100
    r3 = np.mean([z[1] == 3 for z in out]) * 100
    print(f"{lbl:<20}{(vals>0).mean()*100:>7.1f}%{vals.mean():>+8.3f}{vals.sum():>+9.0f}"
          f"{(vals<=-5).mean()*100:>6.1f}%{r3:>7.1f}%{vals.min():>+8.2f}")
