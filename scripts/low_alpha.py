"""저변동 전략의 코스피 대비 알파 (2년)

거래별로 진입일~청산일 시장 수익률을 빼서 초과 수익을 계산한다.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
BUY_GAP, SELL_GAP, FEE = 0.015, 0.020, 0.00015
AUTO_STOP = 0.035
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_AMT, MAX_AMT = 10.0, 37.51
POS_MIN = 89.47
SURGE = 20.0
MONTHS = 24


def cooldown_days():
    m = pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"]).set_index("Date")
    dn = [int(d.strftime("%Y%m%d")) for d in m.index]
    ret = m["Close"].pct_change() * 100
    return {dn[i] for i in range(1, len(dn)) if ret.iloc[i] <= -3.0}, m.index


def trade(o, c, h, l, i):
    e = i + 1
    if e >= len(c):
        return None, None
    prev = float(c[i])
    if float(o[e]) < prev:
        px = float(o[e])
    elif float(l[e]) <= prev:
        px = prev
    else:
        return None, None
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    stop_px = px * (1 - AUTO_STOP)
    limit_entry = abs(px - float(o[e])) > 1e-9
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
            _hit = (h[j] >= held[k] * (1 + SELL_GAP)) if (dd > 0 or not limit_entry) \
                else (c[j] >= held[k] * (1 + SELL_GAP))
            if _hit:
                net += AMTS[k] * 10_000 * (SELL_GAP - FEE)
                del held[k]
        if held and l[j] <= stop_px:
            fill = min(stop_px, float(o[j])) if float(o[j]) < stop_px else stop_px
            for k in list(held):
                net += AMTS[k] * 10_000 * (fill / held[k] - 1 - FEE)
                del held[k]
            return net, j
        if held and c[j] < lv[2]:
            for k in list(held):
                net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
                del held[k]
            return net, j
    if held:
        j = min(e + HOLD - 1, len(c) - 1)
        for k in list(held):
            net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
        return net, j
    return net, e


mkt = pd.read_csv(ROOT / "data" / "ks11_ohlc.csv", parse_dates=["Date"]).set_index("Date")["Close"]
mk = {int(d.strftime("%Y%m%d")): float(v) for d, v in mkt.items()}
mk_days = sorted(mk)

codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
blk, cal = cooldown_days()
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
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
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if not (MIN_AMT <= amt20[i] <= MAX_AMT) or pos < POS_MIN:
            continue
        if (c[i] / c[i - 3] - 1) * 100 > SURGE:
            continue
        d0 = int(df.index[i].strftime("%Y%m%d"))
        if d0 in blk:
            continue
        net, jx = trade(o, c, h, l, i)
        if net is None:
            continue
        d1 = int(df.index[min(jx, n - 1)].strftime("%Y%m%d"))
        trades.append(dict(sig=d0, exit=d1, net=net,
                           pct=net / SLOT * 100))

# 시장 수익률 (신호일 종가 → 청산일 종가)
from bisect import bisect_left


def mret(a, b):
    i = bisect_left(mk_days, a)
    if i >= len(mk_days):
        return 0.0
    j = min(bisect_left(mk_days, b), len(mk_days) - 1)
    return (mk[mk_days[j]] / mk[mk_days[i]] - 1) * 100


for t in trades:
    t["mkt"] = mret(t["sig"], t["exit"])
    t["alpha"] = t["pct"] - t["mkt"]

d = pd.DataFrame(trades)
print(f"[INFO] 거래 {len(d):,}건")
print(f"건당 수익       {d.pct.mean():+.3f}%")
print(f"같은 기간 시장   {d.mkt.mean():+.3f}%")
print(f"알파(초과)      {d.alpha.mean():+.3f}%p")
print(f"알파 플러스 비율 {(d.alpha>0).mean()*100:.1f}%")
print(f"총 수익률 합계  {d.pct.sum():+.0f}% · 시장 합계 {d.mkt.sum():+.0f}% · 알파 합계 {d.alpha.sum():+.0f}%p")

# 월별
d["m"] = d.sig // 100
rows = []
for m, g in d.groupby("m"):
    rows.append(dict(month=f"{m//100}-{m%100:02d}", n=len(g),
                     ret=g.pct.mean(), mkt=g.mkt.mean(), alpha=g.alpha.mean()))
r = pd.DataFrame(rows)
print(f"\n{'월':<9}{'건수':>6}{'건당%':>9}{'시장%':>9}{'알파%p':>9}")
for _, x in r.iterrows():
    print(f"{x.month:<9}{x.n:>6}{x.ret:>+9.3f}{x.mkt:>+9.3f}{x.alpha:>+9.3f}")
print(f"\n알파 플러스 {int((r.alpha>0).sum())}/{len(r)}개월 · 월 평균 알파 {r.alpha.mean():+.3f}%p")
