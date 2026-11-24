"""급등 필터를 자동매도(-3.5%) 체제에서 다시 본다

현행: 3일 수익률 20% 초과 제외 + 당일 15% 초과 제외
질문: 자동매도가 있으면 급등 종목을 넣어도 되는가
"""
import json
from bisect import bisect_left
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
MONTHS = 24


def trade(o, c, h, l, i):
    """(순익 원, 청산 인덱스, 청산 사유)"""
    e = i + 1
    if e >= len(c):
        return None, None, None
    prev = float(c[i])
    if float(o[e]) < prev:
        px = float(o[e])
    elif float(l[e]) <= prev:
        px = prev
    else:
        return None, None, None
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
            return net, j, "자동손절"
        if held and c[j] < lv[2]:
            for k in list(held):
                net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
            return net, j, "3차이탈"
    if held:
        j = min(e + HOLD - 1, len(c) - 1)
        for k in list(held):
            net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
        return net, j, "만기"
    return net, e, "전량익절"


mk = pd.read_csv(ROOT / "data" / "ks11_ohlc.csv", parse_dates=["Date"]).set_index("Date")["Close"]
mkd = {int(d.strftime("%Y%m%d")): float(v) for d, v in mk.items()}
mk_days = sorted(mkd)


def mret(a, b):
    i = bisect_left(mk_days, a)
    if i >= len(mk_days):
        return 0.0
    j = min(bisect_left(mk_days, b), len(mk_days) - 1)
    return (mkd[mk_days[j]] / mkd[mk_days[i]] - 1) * 100


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
    n = len(df)
    cs = pd.Series(c)
    lo30 = cs.rolling(30).min().values; hi30 = cs.rolling(30).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    for i in range(270, n - 1):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        if not (MIN_AMT <= amt20[i] <= MAX_AMT):
            continue
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if pos < POS_MIN:
            continue
        r3 = (c[i] / c[i - 3] - 1) * 100
        r1 = (c[i] / c[i - 1] - 1) * 100
        net, jx, kind = trade(o, c, h, l, i)
        if net is None:
            continue
        d0 = int(df.index[i].strftime("%Y%m%d"))
        d1 = int(df.index[min(jx, n - 1)].strftime("%Y%m%d"))
        rows.append(dict(sig=d0, r3=r3, r1=r1, net=net,
                         pct=net / SLOT * 100, mkt=mret(d0, d1), kind=kind))

d = pd.DataFrame(rows)
print(f"[INFO] 기본 조건 통과 {len(d):,}건 (급등 필터 없음)\n")

VAR = [
    ("필터 없음", lambda x: x.index == x.index),
    ("3일 20% (현행)", lambda x: x.r3 <= 20),
    ("3일 20% + 당일 15% (현행 전체)", lambda x: (x.r3 <= 20) & (x.r1 <= 15)),
    ("3일 20% + 당일 20%", lambda x: (x.r3 <= 20) & (x.r1 <= 20)),
    ("3일 30%", lambda x: x.r3 <= 30),
    ("3일 40%", lambda x: x.r3 <= 40),
    ("3일 50%", lambda x: x.r3 <= 50),
    ("당일 15%만", lambda x: x.r1 <= 15),
    ("3일 10%", lambda x: x.r3 <= 10),
    ("3일 30% + 당일 15%", lambda x: (x.r3 <= 30) & (x.r1 <= 15)),
    ("3일 40% + 당일 15%", lambda x: (x.r3 <= 40) & (x.r1 <= 15)),
    ("3일 40% + 당일 20%", lambda x: (x.r3 <= 40) & (x.r1 <= 20)),
    ("3일 50% + 당일 20%", lambda x: (x.r3 <= 50) & (x.r1 <= 20)),
]
print(f"{'필터':<28}{'건수':>7}{'승률':>7}{'건당%':>8}{'알파%p':>8}{'자동손절':>8}{'합계(원)':>12}")
for name, f in VAR:
    x = d[f(d)]
    if not len(x):
        continue
    a = x.pct - x.mkt
    print(f"{name:<28}{len(x):>7,}{(x.net>0).mean()*100:>6.1f}%{x.pct.mean():>+8.3f}"
          f"{a.mean():>+8.3f}{(x.kind=='자동손절').mean()*100:>7.1f}%{x.net.sum():>+12,.0f}")

print("\n[급등 종목만 따로] 3일 수익률 구간별 (자동매도 체제)")
print(f"{'3일 수익률':<16}{'건수':>7}{'승률':>7}{'건당%':>8}{'알파%p':>8}{'자동손절':>8}")
for lo, hi, lab in ((20, 30, "20~30%"), (30, 50, "30~50%"), (50, 9999, "50% 초과"),
                    (-9999, 0, "하락(-)"), (0, 10, "0~10%")):
    x = d[(d.r3 > lo) & (d.r3 <= hi)]
    if not len(x):
        continue
    a = x.pct - x.mkt
    print(f"{lab:<16}{len(x):>7,}{(x.net>0).mean()*100:>6.1f}%{x.pct.mean():>+8.3f}"
          f"{a.mean():>+8.3f}{(x.kind=='자동손절').mean()*100:>7.1f}%")
