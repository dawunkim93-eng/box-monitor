"""어떤 조건이 3차 체결까지 가는지 (2년)

특징별로 3차 체결률과 그때의 성과를 본다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
COND = ROOT / "data" / "low_cond.json"
FEE = 0.00015
BUY_GAP, SELL_GAP = 0.015, 0.020
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_AMT, MAX_AMT, POS_MIN, SURGE = 10.0, 37.51, 89.47, 20.0
MONTHS = 24

bands = json.loads(COND.read_text(encoding="utf-8"))
CELLS = {(c["amt"], c["pos"], c["atr"]): c["score"] for c in bands["cells"]}
B = bands["bands"]


def cell_score(amt, pos, atr):
    def idx(v, bs):
        for i, (lo, hi) in enumerate(bs):
            if lo <= v < hi:
                return i
        return None
    ai, pi, ti = idx(amt, B["amt"]), idx(pos, B["pos"]), idx(atr, B["atr"])
    if None in (ai, pi, ti):
        return 0.0
    key = (f"{B['amt'][ai][0]:.0f}~{B['amt'][ai][1]:.0f}억",
           f"{B['pos'][pi][0]:.0f}~{min(B['pos'][pi][1],100):.0f}%",
           f"{B['atr'][ti][0]:.1f}~{min(B['atr'][ti][1],20):.0f}%")
    return CELLS.get(key, 0.0)


def run(o, c, h, l, e, px):
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    tg = [x * (1 + SELL_GAP) for x in lv]
    held, net, rungs = {0: 0}, 0.0, 1
    for dd in range(HOLD):
        j = e + dd
        if j >= len(c):
            break
        for k in (1, 2):
            if k not in held and l[j] <= lv[k]:
                held[k] = k
                rungs = max(rungs, k + 1)
        for k in list(held):
            if h[j] >= tg[k]:
                net += AMTS[k] * 10_000 * (SELL_GAP - FEE)
                del held[k]
        if held and c[j] < lv[2]:
            for k in list(held):
                net += AMTS[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
            return net, rungs
        if not held:
            return net, rungs
    j = min(e + HOLD - 1, len(c) - 1)
    for k in list(held):
        net += AMTS[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
    return net, rungs


codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
recs = []
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
    tr = np.concatenate([[h[0] - l[0]],
                         np.maximum(h[1:] - l[1:],
                                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))])
    atr = pd.Series(tr).ewm(alpha=1 / 14, min_periods=14).mean().values / np.maximum(c, 1e-9) * 100
    cs = pd.Series(c)
    lo30 = cs.rolling(30).min().values; hi30 = cs.rolling(30).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    for i in range(270, n - 2):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        a = amt20[i]
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if not (MIN_AMT <= a <= MAX_AMT) or pos < POS_MIN or np.isnan(atr[i]):
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
        net, rungs = run(o, c, h, l, e, px)
        recs.append(dict(amt=a, pos=pos, atr=atr[i], ret3=(c[i] / c[i - 3] - 1) * 100,
                         score=cell_score(a, pos, atr[i]),
                         net=net / SLOT * 100, rungs=rungs))

d = pd.DataFrame(recs)
print(f"[INFO] {len(d):,}건 · 3차 체결 {int((d.rungs==3).mean()*100)}%")


def table(col, cuts, labels):
    print(f"\n[{col}]")
    print(f"{'구간':<16}{'건수':>7}{'3차체결':>8}{'승률':>7}{'건당%':>8}{'합계%':>9}{'-5%↓':>7}")
    for (lo, hi), lab in zip(cuts, labels):
        s = d[(d[col] >= lo) & (d[col] < hi)]
        if not len(s):
            continue
        print(f"{lab:<16}{len(s):>7,}{(s.rungs==3).mean()*100:>7.1f}%"
              f"{(s.net>0).mean()*100:>7.1f}%{s.net.mean():>+8.3f}{s.net.sum():>+9.0f}"
              f"{(s.net<=-5).mean()*100:>6.1f}%")


table("atr", [(0, 3), (3, 4), (4, 5), (5, 7), (7, 25)],
      ["ATR 3% 미만", "3~4%", "4~5%", "5~7%", "7% 이상"])
table("amt", [(10, 15), (15, 20), (20, 25), (25, 31), (31, 38)],
      ["거래대금 10~15억", "15~20억", "20~25억", "25~31억", "31~38억"])
table("pos", [(89, 95), (95, 98), (98, 100), (100, 101)],
      ["위치 89~95%", "95~98%", "98~100%", "100% (신고가)"])
table("ret3", [(-30, 0), (0, 5), (5, 10), (10, 15), (15, 21)],
      ["3일 -30~0%", "0~5%", "5~10%", "10~15%", "15~20%"])
table("score", [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)],
      ["점수 20 미만", "20~40", "40~60", "60~80", "80 이상"])
