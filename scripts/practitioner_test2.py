"""실무자 방식 재검증 — 청산 규칙을 바꿔서 (추세 추종은 시간이 필요)

대상: 1) 150일선 돌파 + 거래량 1.5배, 2) 미너비니 추세 템플릿
청산: (10%/-8%/20일), (20%/-10%/60일), (목표 없이 60일 보유)
추가: 거래대금 구간 분해
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
FEE = 0.00015
MONTHS = 24


def sim(o, h, l, c, e, tgt, stop, maxd):
    px = float(o[e])
    if px <= 0:
        return None
    for d in range(maxd):
        j = e + d
        if j >= len(c):
            return None
        if tgt > 0 and h[j] >= px * (1 + tgt):
            return tgt - FEE
        if stop > 0 and l[j] <= px * (1 - stop):
            return -stop - FEE
    j = min(e + maxd - 1, len(c) - 1)
    return float(c[j]) / px - 1 - FEE


codes = sorted(set(json.loads((ROOT / "data" / "daily_universe.json").read_text(encoding="utf-8"))["codes"]))
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
    if len(df) < 330:
        continue
    o = df["Open"].values.astype(float)
    c = df["Close"].values.astype(float); h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float); v = df["Volume"].values.astype(float)
    n = len(df)
    cs = pd.Series(c)
    ma150 = cs.rolling(150).mean().values
    ma200 = cs.rolling(200).mean().values
    v50 = pd.Series(v).rolling(50).mean().values
    lo250 = cs.rolling(250).min().values
    hi250 = cs.rolling(250).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    prev = False
    for i in range(255, n - 62):
        if df.index[i] < start or amt20[i] < 5.0 or v50[i] <= 0:
            continue
        w = (ma150[i - 1] == ma150[i - 1] and c[i - 1] <= ma150[i - 1]
             and c[i] > ma150[i] and v[i] >= 1.5 * v50[i])
        m = (ma200[i] == ma200[i] and ma150[i] == ma150[i]
             and c[i] > ma150[i] and c[i] > ma200[i] and ma150[i] > ma200[i]
             and ma200[i] > ma200[i - 20]
             and lo250[i] > 0 and c[i] >= lo250[i] * 1.25
             and hi250[i] > 0 and c[i] >= hi250[i] * 0.75)
        if not (w or m):
            prev = m
            continue
        if m and prev:          # 이미 통과 중이면 신규 신호 아님
            prev = m
            continue
        prev = m
        e = i + 1
        for tag, tgt, stop, md in (("a", 0.10, 0.08, 20), ("b", 0.20, 0.10, 60), ("c", 0.0, 0.0, 60)):
            r = sim(o, h, l, c, e, tgt, stop, md)
            if r is None:
                continue
            rows.append(dict(strat=("wein" if w else "miner"), tag=tag, net=r * 100, amt=amt20[i]))

d = pd.DataFrame(rows)
print(f"{'방식':<12}{'청산':<18}{'건수':>7}{'승률':>7}{'건당%':>8}")
lab = {"a": "10%/-8%/20일", "b": "20%/-10%/60일", "c": "목표없이 60일"}
for s, nm in (("wein", "150일선 돌파"), ("miner", "미너비니")):
    for tag in ("a", "b", "c"):
        x = d[(d.strat == s) & (d.tag == tag)].net.values
        if not len(x):
            continue
        print(f"{nm:<12}{lab[tag]:<18}{len(x):>7,}{(x>0).mean()*100:>7.1f}%{x.mean():>+8.3f}")

print(f"\n미너비니 · 60일 보유 · 거래대금 구간별")
sub = d[(d.strat == "miner") & (d.tag == "c")]
for lo, hi, t in ((0, 10, "10억 이하"), (10, 20, "10~20억"), (20, 50, "20~50억"), (50, 1e9, "50억 초과")):
    x = sub[(sub.amt >= lo) & (sub.amt < hi)].net.values
    if not len(x):
        continue
    print(f"   {t:<12}{len(x):>7,}{(x>0).mean()*100:>7.1f}%{x.mean():>+8.3f}")
