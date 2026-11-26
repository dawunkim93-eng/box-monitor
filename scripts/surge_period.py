"""3일 급등 필터 비교 (기간별) — 11/26까지 반영

필터: 3일 20% / 3일 30% / 3일 40%(현행) / 제한 없음
기간: 전체 2년 / 최근 3개월(9월~) / 최근 1개월(11월)
슬리피지: 매수 -0.05%, 익절 0, 자동손절 -0.3%
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
SLIP_BUY = float(os.environ.get("SLIP_BUY", "0.05")) / 100
SLIP_STOP = float(os.environ.get("SLIP_STOP", "0.3")) / 100
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_AMT, MAX_AMT, POS_MIN = 10.0, 37.51, 89.47
SURGE1 = 20.0
MONTHS = 24


def trade(o, c, h, l, i):
    e = i + 1
    if e >= len(c):
        return None
    prev = float(c[i])
    if float(o[e]) < prev:
        px = float(o[e])
    elif float(l[e]) <= prev:
        px = prev
    else:
        return None
    px *= (1 + SLIP_BUY)
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    stop_px = px * (1 - AUTO_STOP)
    limit_entry = abs(px - float(o[e]) * (1 + SLIP_BUY)) < 1e-6
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
    cs = pd.Series(c)
    lo30 = cs.rolling(30).min().values; hi30 = cs.rolling(30).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    days = [int(x.strftime("%Y%m%d")) for x in df.index]
    for i in range(270, len(df) - 1):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        if not (MIN_AMT <= amt20[i] <= MAX_AMT):
            continue
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if pos < POS_MIN:
            continue
        r3 = (c[i] / c[i - 3] - 1) * 100
        r1 = (c[i] / c[i - 1] - 1) * 100
        if r1 > SURGE1:                     # 당일 급등은 항상 제외
            continue
        net = trade(o, c, h, l, i)
        if net is None:
            continue
        rows.append(dict(sig=days[i], r3=r3, net=net))

d = pd.DataFrame(rows)
d["m"] = d.sig // 100
print(f"[INFO] 신호 {len(d):,}건 (당일 20% 제외 적용)")

PERIODS = [("전체 2년", 0), ("최근 3개월", 202609), ("최근 1개월", 202611)]
FILTERS = [("제한 없음", 1e9), ("3일 40% (현행)", 40.0), ("3일 30%", 30.0), ("3일 20%", 20.0)]

for pname, from_m in PERIODS:
    sub = d[d.m >= from_m] if from_m else d
    print(f"\n[{pname}] {len(sub):,}건")
    print(f"  {'필터':<16}{'건수':>7}{'승률':>8}{'건당%':>9}{'합계(원)':>13}")
    for fname, thr in FILTERS:
        x = sub[sub.r3 <= thr].net.values
        if not len(x):
            continue
        pct = x / SLOT * 100
        print(f"  {fname:<16}{len(x):>7,}{(x>0).mean()*100:>7.1f}%{pct.mean():>+9.3f}{x.sum():>+13,.0f}")
