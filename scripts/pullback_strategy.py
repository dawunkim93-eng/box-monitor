"""재료 급등 후 조정 매수 검증 (2년)

1) 재료일: 하루 +10% 이상 & 거래량 20일 평균의 5배 이상
2) 조정: 이후 2~10거래일 안에 고점 대비 -8% 이상 밀리고 20일선 부근까지 내려옴
3) 매수: 조정 중 종가가 전일보다 오른 날(반등 확인) 다음날 시가
4) 청산: 목표/손절/최대 보유
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
MIN_AMT = 5.0


def sim(o, h, l, c, e, tgt, stop, maxd):
    px = float(o[e])
    if px <= 0:
        return None
    for d in range(maxd):
        j = e + d
        if j >= len(c):
            return None
        if h[j] >= px * (1 + tgt):
            return tgt - FEE
        if l[j] <= px * (1 - stop):
            return -stop - FEE
    j = min(e + maxd - 1, len(c) - 1)
    return float(c[j]) / px - 1 - FEE


codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
sig = []
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
    ma20 = cs.rolling(20).mean().values
    v20 = pd.Series(v).rolling(20).mean().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    for i in range(270, n - 12):
        if df.index[i] < start or amt20[i] < MIN_AMT:
            continue
        if v20[i] <= 0 or c[i - 1] <= 0:
            continue
        up = (c[i] / c[i - 1] - 1) * 100
        vr = v[i] / v20[i]
        if up < 10 or vr < 5:
            continue                        # 재료일
        pk = c[i]
        for j in range(i + 2, min(i + 11, n - 1)):
            if c[j] > pk * 1.05:
                pk = c[j]                     # 고점 갱신
            dd = (c[j] / pk - 1) * 100
            near20 = abs(c[j] / ma20[j] - 1) <= 0.05 if ma20[j] > 0 else False
            rebound = c[j] > c[j - 1]
            if dd <= -8 and (near20 or dd <= -15) and rebound:
                sig.append((o, h, l, c, j + 1, up, vr))
                break
print(f"[INFO] 신호 {len(sig):,}건 · 종목 {len(set(id(s[0]) for s in sig)):,}")
print(f"\n{'목표/손절/보유':<20}{'건수':>7}{'승률':>7}{'건당%':>8}{'합계%':>9}")
for tgt, stop, maxd in [(0.10, 0.08, 15), (0.08, 0.08, 10), (0.10, 0.10, 20),
                         (0.15, 0.10, 20), (0.05, 0.05, 10), (0.20, 0.10, 30)]:
    out = [sim(o, h, l, c, e, tgt, stop, maxd) for o, h, l, c, e, _, _ in sig]
    out = [x for x in out if x is not None]
    r = np.array(out) * 100
    print(f"{tgt*100:.0f}% / -{stop*100:.0f}% / {maxd}일{'':<4}{len(r):>7,}"
          f"{(r>0).mean()*100:>7.1f}%{r.mean():>+8.3f}{r.sum():>+9.0f}")
