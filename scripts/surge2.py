"""급등 필터 재검증 (2년, 아침 매수·매도 +2.0%·자동손절 -3.5%·진입일 종가 판정)

현행: 3일 수익률 20% 초과 제외
대안: 3일 15% / 당일 상승률 조건 추가
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
FEE = 0.00015
BUY_GAP, SELL_GAP, AUTO_STOP = 0.015, 0.020, 0.035
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_AMT, MAX_AMT, POS_MIN = 10.0, 37.51, 89.47
MONTHS = 24


def trade(o, c, h, l, e, px):
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    tg = [x * (1 + SELL_GAP) for x in lv]
    stop_px = px * (1 - AUTO_STOP)
    limit_entry = abs(px - float(o[e])) > 1e-9
    held = {0: 0}
    net = 0.0
    for dd in range(HOLD):
        j = e + dd
        if j >= len(c):
            break
        for k in (1, 2):
            if k not in held and l[j] <= lv[k]:
                held[k] = k
        for k in list(held):
            hit = (h[j] >= tg[k]) if (dd > 0 or not limit_entry) else (c[j] >= tg[k])
            if hit:
                net += AMTS[k] * 10_000 * (SELL_GAP - FEE)
                del held[k]
        if held and l[j] <= stop_px:
            fill = min(stop_px, float(o[j])) if float(o[j]) < stop_px else stop_px
            for k in list(held):
                net += AMTS[k] * 10_000 * (fill / lv[k] - 1 - FEE)
            return net
        if held and c[j] < lv[2]:
            for k in list(held):
                net += AMTS[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
            return net
        if not held:
            return net
    j = min(e + HOLD - 1, len(c) - 1)
    for k in list(held):
        net += AMTS[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
    return net


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
    lo30 = cs.rolling(30).min().values; hi30 = cs.rolling(30).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    for i in range(270, n - 2):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        a = amt20[i]
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if not (MIN_AMT <= a <= MAX_AMT) or pos < POS_MIN:
            continue
        e = i + 1
        prev = float(c[i])
        if float(o[e]) < prev:
            px = float(o[e])
        elif float(l[e]) <= prev:
            px = prev
        else:
            continue
        rows.append(dict(ret3=(c[i] / c[i - 3] - 1) * 100,
                         ret1=(c[i] / c[i - 1] - 1) * 100,
                         ret5=(c[i] / c[i - 5] - 1) * 100,
                         net=trade(o, c, h, l, e, px) / SLOT * 100))

d = pd.DataFrame(rows)
print(f"[INFO] 신호 {len(d):,}건 · 현행(3일 20%) 통과 {int((d.ret3<=20).sum()):,}건")


def show(mask, label):
    x = d[mask].net.values
    print(f"{label:<26}{len(x):>8,}{(x>0).mean()*100:>7.1f}%{x.mean():>+8.3f}"
          f"{x.sum():>+9.0f}{(x<=-5).mean()*100:>6.1f}%")


print(f"\n{'필터':<26}{'건수':>8}{'승률':>7}{'건당%':>8}{'합계%':>9}{'-5%↓':>7}")
show(d.ret3 <= 20, "현행 3일 20%")
show(d.ret3 <= 15, "3일 15%")
show((d.ret3 <= 20) & (d.ret1 <= 10), "3일 20% + 당일 10%")
show((d.ret3 <= 20) & (d.ret1 <= 15), "3일 20% + 당일 15%")
show((d.ret3 <= 20) & (d.ret5 <= 25), "3일 20% + 5일 25%")
show(d.ret5 <= 20, "5일 20%")
