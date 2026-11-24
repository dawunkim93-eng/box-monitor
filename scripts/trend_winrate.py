"""추세 전환 승률 높이기 — 조건 추가별 효과 (2년)

기본: 골든크로스(5일 내) + 거래량 1.5배 + 종가>20일선 + 거래대금 10억+ + 60일최저 대비 +5~25%
청산: 목표 10% / 손절 10% / 20일
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
FEE = 0.00015
MONTHS = 24


def simulate(o, c, h, l, e, tgt=0.10, stop=0.10, maxd=20):
    px = float(o[e])
    if px <= 0:
        return None
    for d in range(maxd):
        j = e + d
        if j >= len(c):
            return None
        if h[j] >= px * (1 + tgt):
            return tgt - FEE, d + 1
        if l[j] <= px * (1 - stop):
            return -stop - FEE, d + 1
    j = min(e + maxd - 1, len(c) - 1)
    return float(c[j]) / px - 1 - FEE, maxd


mk = pd.read_csv(ROOT / "data" / "ks11_ohlc.csv", parse_dates=["Date"]).set_index("Date")
mk_c = mk["Close"]
mk_ret60 = (mk_c / mk_c.shift(60) - 1) * 100
mk_map = {int(d.strftime("%Y%m%d")): float(v) for d, v in mk_ret60.items()}

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
    ma5 = cs.rolling(5).mean().values
    ma20 = cs.rolling(20).mean().values
    ma60 = cs.rolling(60).mean().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    vol5 = pd.Series(v).rolling(5).mean().values
    vol20 = pd.Series(v).rolling(20).mean().values
    lo60 = cs.rolling(60).min().values
    for i in range(270, n - 25):
        if df.index[i] < start or amt20[i] < 10.0:
            continue
        if not (ma5[i] > ma20[i] and c[i] > ma20[i]):
            continue
        if not any(ma5[k] > ma20[k] and ma5[k - 1] <= ma20[k - 1] for k in range(i - 4, i + 1)):
            continue
        vr = vol5[i] / vol20[i] if vol20[i] > 0 else 0
        if vr < 1.5:
            continue
        up = (c[i] / lo60[i] - 1) * 100 if lo60[i] > 0 else 999
        if not (5 < up <= 25):
            continue
        r = simulate(o, c, h, l, i + 1)
        if r is None:
            continue
        d0 = int(df.index[i].strftime("%Y%m%d"))
        rows.append(dict(net=r[0] * 100, days=r[1], vr=vr,
                         slope=ma20[i] > ma20[i - 5],
                         above60=c[i] > ma60[i],
                         rs=(c[i] / c[i - 60] - 1) * 100 - mk_map.get(d0, 0),
                         amt=amt20[i]))

d = pd.DataFrame(rows)
print(f"[INFO] 기본 신호 {len(d):,}건 · 승률 {(d.net>0).mean()*100:.1f}% · 건당 {d.net.mean():+.3f}%")


def show(mask, label):
    x = d[mask]
    if not len(x):
        return
    print(f"{label:<30}{len(x):>6,}{len(x)/len(d)*100:>6.0f}%{(x.net>0).mean()*100:>7.1f}%"
          f"{x.net.mean():>+8.3f}{x.net.sum():>+8.0f}{x.days.mean():>7.1f}")


print(f"\n{'조건':<30}{'건수':>6}{'비중':>6}{'승률':>7}{'건당%':>8}{'합계%':>8}{'보유':>7}")
show(np.ones(len(d), bool), "기본")
show(d.vr >= 2.0, "+ 거래량 2배")
show(d.vr >= 3.0, "+ 거래량 3배")
show(d.above60, "+ 60일선 위")
show(~d.above60, "  60일선 아래")
show(d.slope, "+ 20일선 상승")
show(d.rs > 0, "+ 지수 대비 강함(60일)")
show(d.rs > 10, "+ 지수 대비 +10%p 이상")
show(d.amt <= 20, "+ 거래대금 20억 이하")
show(d.amt > 20, "+ 거래대금 20억 초과")
show((d.vr >= 2.0) & d.above60, "+ 거래량 2배 & 60일선 위")
show((d.vr >= 2.0) & d.above60 & (d.rs > 0), "+ 2배 & 60일선 & 지수강함")
print()
show((d.vr >= 2.0) & (d.amt <= 20), "★ 2배 & 거래대금 20억 이하")
show((d.vr >= 2.0) & (d.amt <= 20) & (~d.above60), "★ 2배 & 20억 이하 & 60일선 아래")
show((d.vr >= 3.0) & (d.amt <= 20), "★ 3배 & 20억 이하")
show((d.vr >= 3.0) & (d.amt <= 20) & (~d.above60), "★ 3배 & 20억 이하 & 60일선 아래")
