"""바닥 추세 전환 차트 식별 검증 (2년)

기존 조건(거래대금 10~37.5억 · 밴드위치 89.5%+) 위에
추세 전환 신호를 하나씩 얹어 성과 차이를 본다.

S1 골든크로스: 5일선이 20일선 위로 최근 5일 내 교차
S2 20일선 상승 전환: 20일선이 5일 전보다 높음
S3 바닥 반등: 60일 최저가 대비 +15% 이상
S4 거래량 증가: 5일 평균 거래량 / 20일 평균 ≥ 1.5
S5 60일 신고가 근접: 60일 최고가의 95% 이상
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
MONTHS = 24


def trade(o, c, h, l, e, px):
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    tg = [x * (1 + SELL_GAP) for x in lv]
    stop_px = px * (1 - AUTO_STOP)
    limit_entry = abs(px - float(o[e])) > 1e-9
    held, net = {0: 0}, 0.0
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
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=24)
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
    ma5 = cs.rolling(5).mean().values
    ma20 = cs.rolling(20).mean().values
    lo60 = cs.rolling(60).min().values
    hi60 = cs.rolling(60).max().values
    vol5 = pd.Series(v).rolling(5).mean().values
    vol20 = pd.Series(v).rolling(20).mean().values
    for i in range(270, n - 2):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        a = amt20[i]
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if not (10.0 <= a <= 37.51) or pos < 89.47:
            continue
        if (c[i] / c[i - 1] - 1) * 100 > 15 or (c[i] / c[i - 3] - 1) * 100 > 20:
            continue
        e = i + 1
        prev = float(c[i])
        if float(o[e]) < prev:
            px = float(o[e])
        elif float(l[e]) <= prev:
            px = prev
        else:
            continue
        s1 = any(ma5[k] > ma20[k] and ma5[k - 1] <= ma20[k - 1] for k in range(i - 4, i + 1))
        s2 = ma20[i] > ma20[i - 5]
        s3 = c[i] >= lo60[i] * 1.15
        s4 = vol20[i] > 0 and vol5[i] / vol20[i] >= 1.5
        s5 = c[i] >= hi60[i] * 0.95
        rows.append(dict(net=trade(o, c, h, l, e, px) / SLOT * 100,
                         s1=s1, s2=s2, s3=s3, s4=s4, s5=s5))

d = pd.DataFrame(rows)
print(f"[INFO] 조건 충족 {len(d):,}건 · 건당 {d.net.mean():+.3f}% · 승률 {(d.net>0).mean()*100:.1f}%")


def show(mask, label):
    x = d[mask].net.values
    if not len(x):
        return
    print(f"{label:<28}{len(x):>7,}{len(x)/len(d)*100:>6.0f}%{(x>0).mean()*100:>7.1f}%"
          f"{x.mean():>+8.3f}{x.sum():>+9.0f}")


print(f"\n{'조건':<28}{'건수':>7}{'비중':>6}{'승률':>7}{'건당%':>8}{'합계%':>9}")
show(np.ones(len(d), bool), "전체")
for col, lbl in (("s1", "S1 골든크로스(5일내)"), ("s2", "S2 20일선 상승전환"),
                 ("s3", "S3 바닥대비 +15%"), ("s4", "S4 거래량 1.5배"),
                 ("s5", "S5 60일 신고가 95%")):
    show(d[col].values, lbl)
print()
for col, lbl in (("s1", "S1"), ("s2", "S2"), ("s4", "S4"), ("s5", "S5")):
    show(~d[col].values, lbl + " 아닌 것")
