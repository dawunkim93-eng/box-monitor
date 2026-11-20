"""아침 매수 기준 쿨다운 효과 검증 (2년)

A 없음 / B 신호일 코스피 -3% 이하 차단 / C 매수일 시가 갭 -1% 이하 차단
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
GAP, FEE = 0.015, 0.00015
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_AMT, MAX_AMT, POS_MIN, SURGE = 10.0, 37.51, 89.47, 20.0
MONTHS = 24


def mkt():
    m = pd.read_csv(ROOT / "data" / "ks11_ohlc.csv", parse_dates=["Date"]).set_index("Date")
    dn = [int(d.strftime("%Y%m%d")) for d in m.index]
    c = m["Close"].values.astype(float)
    o = m["Open"].values.astype(float)
    ret = np.concatenate([[0.0], c[1:] / c[:-1] * 100 - 100])
    gap = o / np.concatenate([[o[0]], c[:-1]]) * 100 - 100
    return {dn[i]: ret[i] for i in range(len(dn))}, {dn[i]: gap[i] for i in range(len(dn))}, dn


def run_morning(o, c, h, l, i):
    e = i + 1
    if e >= len(c):
        return None
    prev = c[i]
    if o[e] < prev:
        px = o[e]
    elif l[e] <= prev:
        px = prev
    else:
        return None                      # 미체결
    lv = [px * (1 - GAP) ** k for k in range(3)]
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
            if h[j] >= held[k] * (1 + GAP):
                net += AMTS[k] * 10_000 * (GAP - FEE)
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
    return net


codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
ret_map, gap_map, cal = mkt()
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
calib = sorted({d for d in cal if d >= int(start.strftime("%Y%m%d"))})
nextday = {calib[i]: calib[i + 1] for i in range(len(calib) - 1)}
trades = []

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
    for i in range(270, n - 1):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        a = amt20[i]
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if not (MIN_AMT <= a <= MAX_AMT) or pos < POS_MIN:
            continue
        if (c[i] / c[i - 3] - 1) * 100 > SURGE:
            continue
        d0 = int(df.index[i].strftime("%Y%m%d"))
        e0 = int(df.index[i + 1].strftime("%Y%m%d"))
        net = run_morning(o, c, h, l, i)
        if net is None:
            continue
        trades.append(dict(sig=d0, entry=e0, net=net,
                           sig_dn=ret_map.get(d0, 0.0),
                           ent_gap=gap_map.get(e0, 0.0)))

a = np.array([t["net"] for t in trades]) / SLOT * 100

def show(mask, label):
    x = a[mask]
    print(f"{label:<26}{len(x):>8,}{(x>0).mean()*100:>7.1f}%{x.mean():>+9.3f}"
          f"{x.sum():>+9.0f}{(x<=-5).mean()*100:>6.1f}%")

print(f"[INFO] 아침 매수 체결 {len(a):,}건")
print(f"\n{'구분':<26}{'건수':>8}{'승률':>7}{'건당%':>9}{'합계%':>9}{'-5%↓':>7}")
sig_dn = np.array([t["sig_dn"] for t in trades])
ent_gap = np.array([t["ent_gap"] for t in trades])
show(np.ones(len(a), bool), "A 쿨다운 없음")
show(sig_dn > -3.0, "B 신호일 -3% 차단")
show(ent_gap > -1.0, "C 매수일 시가 -1% 차단")
show(ent_gap > -2.0, "C' 매수일 시가 -2% 차단")

print(f"\n[차단일 당일 성과]")
show(sig_dn <= -3.0, "신호일 급락일에 잡힌 것")
show(ent_gap <= -1.0, "매수일 시가 -1% 이하")
show(ent_gap <= -2.0, "매수일 시가 -2% 이하")
