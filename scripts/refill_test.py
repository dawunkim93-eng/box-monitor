"""재체결(재그리드) 반영 검증 — 2년

앱 동작: 차수가 매도돼도 주문이 남아, 가격이 그 차수 아래로 다시 내려오면 재매수.
현재 모델: 한 차수는 한 번만 체결(재매수 없음).
비교: 3거래일 동안 재체결 허용 (자동손절 1차 -3.5%는 그대로 종료 조건)
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
AUTO_STOP = 0.035
MIN_AMT, MAX_AMT, POS_MIN, SURGE = 10.0, 37.51, 89.47, 20.0
MONTHS = 24


def trade(o, c, h, l, e, px, refill):
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    tg = [x * (1 + SELL_GAP) for x in lv]
    stop_px = px * (1 - AUTO_STOP)
    held = {0: 0}
    ever = {0}
    net = 0.0
    cycles = 0
    for dd in range(HOLD):
        j = e + dd
        if j >= len(c):
            break
        # 매수 체결 (재체결 포함)
        for k in (1, 2):
            if k in held:
                continue
            if (not refill) and k in ever:
                continue                      # 재체결 금지 모드
            if l[j] <= lv[k]:
                held[k] = k
                ever.add(k)
        # 매도
        for k in list(held):
            if h[j] >= tg[k]:
                net += AMTS[k] * 10_000 * (SELL_GAP - FEE)
                del held[k]
                cycles += 1
        # 자동손절
        if held and l[j] <= stop_px:
            fill = min(stop_px, float(o[j])) if float(o[j]) < stop_px else stop_px
            for k in list(held):
                net += AMTS[k] * 10_000 * (fill / lv[k] - 1 - FEE)
            return net, cycles
        # 종가 3차 이탈
        if held and c[j] < lv[2]:
            for k in list(held):
                net += AMTS[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
            return net, cycles
    j = min(e + HOLD - 1, len(c) - 1)
    for k in list(held):
        net += AMTS[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
    return net, cycles


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
print(f"\n{'방식':<16}{'승률':>7}{'건당%':>8}{'합계%':>9}{'-5%↓':>7}{'평균매도횟수':>12}{'최악%':>8}")
for refill, lbl in ((False, "현재(재체결 없음)"), (True, "재체결 허용")):
    out = [trade(*packs[p], e, x, refill) for p, e, x in sigs]
    vals = np.array([z[0] for z in out]) / SLOT * 100
    cyc = np.mean([z[1] for z in out])
    print(f"{lbl:<16}{(vals>0).mean()*100:>7.1f}%{vals.mean():>+8.3f}{vals.sum():>+9.0f}"
          f"{(vals<=-5).mean()*100:>6.1f}%{cyc:>12.2f}{vals.min():>+8.2f}")
