"""나스닥 -3% 급락 다음날, 매수 시점(장초반 vs 장후반) 비교

장초반 = 시가 매수 / 장후반 = 종가 매수
사다리·자동매도(-3.5%)·만기 3일은 동일하게 적용.
"""
import csv
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
SURGE, SURGE1 = 40.0, 20.0
MONTHS = 24
NDQ_DROP = -3.0


def run_trade(o, c, h, l, e, px, days=3, allow_day0_sell=True):
    """e: 진입일 인덱스, px: 체결가"""
    if px <= 0 or e >= len(c):
        return None
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    stop_px = px * (1 - AUTO_STOP)
    held = {0: px}
    net = 0.0
    for dd in range(days):
        j = e + dd
        if j >= len(c):
            break
        for k in (1, 2):
            if k not in held and l[j] <= lv[k]:
                held[k] = lv[k]
        if dd > 0 or allow_day0_sell:
            for k in list(held):
                if h[j] >= held[k] * (1 + SELL_GAP):
                    net += AMTS[k] * 10_000 * (SELL_GAP - FEE)
                    del held[k]
        if held and l[j] <= stop_px:
            fill = min(stop_px, float(o[j])) if float(o[j]) < stop_px else stop_px
            for k in list(held):
                net += AMTS[k] * 10_000 * (fill / held[k] - 1 - FEE)
            return net
        if held and c[j] < lv[2]:
            for k in list(held):
                net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
            return net
    if held:
        j = min(e + days - 1, len(c) - 1)
        for k in list(held):
            net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
    return net


# 나스닥 일별 등락
nd = []
with (ROOT / "data" / "nasdaq.csv").open() as f:
    for row in csv.DictReader(f):
        nd.append((row["Date"].replace("-", ""), float(row["Close"])))
nd.sort()
nd_days = [d for d, _ in nd]
nd_ret = {}
for i in range(1, len(nd)):
    nd_ret[nd[i][0]] = (nd[i][1] / nd[i - 1][1] - 1) * 100


def ndq_prev(day):
    """day(한국 거래일, 정수) 직전 미국장 등락률"""
    day = str(day)
    j = bisect_left(nd_days, day) - 1
    if j < 0:
        return None, None
    return nd_days[j], nd_ret.get(nd_days[j])


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
    days = [int(x.strftime("%Y%m%d")) for x in df.index]
    for i in range(270, n - 2):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        if not (MIN_AMT <= amt20[i] <= MAX_AMT):
            continue
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if pos < POS_MIN:
            continue
        if (c[i] / c[i - 3] - 1) * 100 > SURGE or (c[i] / c[i - 1] - 1) * 100 > SURGE1:
            continue
        e = i + 1                       # 매수일
        buy_day = days[e]
        ud, ur = ndq_prev(buy_day)
        if ur is None:
            continue
        # 현행 규칙: 시가 < 전일종가 → 시장가, 그 외 → 전일종가 지정가(안 닿으면 미체결)
        prev = float(c[i])
        if float(o[e]) < prev:
            a = run_trade(o, c, h, l, e, float(o[e]), HOLD, allow_day0_sell=True)
        elif float(l[e]) <= prev:
            a = run_trade(o, c, h, l, e, prev, HOLD, allow_day0_sell=False)
        else:
            a = None
        b = run_trade(o, c, h, l, e + 1, float(c[e]), HOLD, allow_day0_sell=True)
        b2 = run_trade(o, c, h, l, e, float(c[e]), HOLD, allow_day0_sell=False)   # 같은 창, 진입가만 종가
        if a is None or b is None:
            continue
        rows.append(dict(net_open=a, net_close=b, net_close_same=b2, ndq=ur,
                         intraday=(float(c[e]) / max(float(o[e]), 1e-9) - 1) if float(o[e]) > 0 else 0.0))

d = pd.DataFrame(rows)
print(f"[INFO] 표본 {len(d):,}건 · 나스닥 급락일 매수 {int((d.ndq <= NDQ_DROP).sum()):,}건")

for label, sub in (("전체", d), (f"나스닥 -{abs(NDQ_DROP):.0f}% 이하 다음날", d[d.ndq <= NDQ_DROP]),
                   ("그 외 날", d[d.ndq > NDQ_DROP])):
    if not len(sub):
        continue
    A = sub.net_open.values / SLOT * 100
    B = sub.net_close.values / SLOT * 100
    print(f"\n[{label}] {len(sub):,}건")
    print(f"  장초반(시가) 승률 {(A>0).mean()*100:5.1f}% · 건당 {A.mean():+.3f}% · 합계 {sub.net_open.sum():+,.0f}원")
    print(f"  장후반(종가) 승률 {(B>0).mean()*100:5.1f}% · 건당 {B.mean():+.3f}% · 합계 {sub.net_close.sum():+,.0f}원")
    C = sub.net_close_same.values / SLOT * 100
    print(f"  같은 창·종가진입 승률 {(C>0).mean()*100:5.1f}% · 건당 {C.mean():+.3f}% · 합계 {sub.net_close_same.sum():+,.0f}원")
    print(f"  차이(종가-시가) 건당 {B.mean()-A.mean():+.3f}%p · 같은 창 기준 {C.mean()-A.mean():+.3f}%p")

# 나스닥 등락 구간별 (같은 보유 구간 기준)
print(f"\n{'나스닥 전일':<14}{'건수':>7}{'시가 건당%':>11}{'같은구간 종가':>13}{'차이%p':>9}{'당일 시가→종가':>14}")
for lo, hi, lab in ((-99, -3, "-3% 이하"), (-3, -2, "-3~-2%"), (-2, -1.5, "-2~-1.5%"),
                    (-1.5, 0, "-1.5~0%"), (0, 1.5, "0~+1.5%"), (1.5, 99, "+1.5% 이상")):
    x = d[(d.ndq > lo) & (d.ndq <= hi)]
    if not len(x):
        continue
    A = x.net_open.values / SLOT * 100
    C = x.net_close_same.values / SLOT * 100
    intra = x.intraday.values * 100
    print(f"{lab:<14}{len(x):>7,}{A.mean():>+11.3f}{C.mean():>+13.3f}{C.mean()-A.mean():>+9.3f}{intra.mean():>+13.2f}%")

# 임계값 2% / 2.5% / 3%
print(f"\n{'임계값':<12}{'건수':>7}{'시가 승률':>10}{'시가 건당%':>12}{'종가 건당%':>12}{'차이%p':>9}")
for thr in (-2.0, -2.5, -3.0):
    x = d[d.ndq <= thr]
    if not len(x):
        continue
    A = x.net_open.values / SLOT * 100
    B = x.net_close.values / SLOT * 100        # 종가 매수 후 3거래일 보유
    print(f"나스닥 {thr:.1f}%↓{'':<2}{len(x):>7,}{(A>0).mean()*100:>9.1f}%{A.mean():>+12.3f}{B.mean():>+12.3f}{B.mean()-A.mean():>+9.3f}"
          f"   (종가 승률 {(B>0).mean()*100:.1f}%)")
