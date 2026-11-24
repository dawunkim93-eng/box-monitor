"""실무자 방식 4종 검증 (2년, 982종목)

1) 와인스타인: 150일선 상향 돌파 + 거래량 1.5배
2) 미너비니 추세 템플릿 8조건 통과 첫날
3) VCP: 최근 60일 변동폭 축소 + 20일 신고가 돌파
4) 오닐: 20일 신고가 돌파 시 거래량 증가율 구간별
청산 공통: 목표 +10% / 손절 -8% / 20일
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
FEE = 0.00015
MONTHS = 24


def sim(o, h, l, c, e, tgt=0.10, stop=0.08, maxd=20):
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


codes = sorted(set(json.loads((ROOT / "data" / "daily_universe.json").read_text(encoding="utf-8"))["codes"]))
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
R = {k: [] for k in ("wein", "wein_up", "miner", "vcp")}
VOL = []
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
    hi20 = pd.Series(h).rolling(20).max().values
    lo250 = cs.rolling(250).min().values
    hi250 = cs.rolling(250).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    miner_prev = False
    for i in range(255, n - 22):
        if df.index[i] < start or amt20[i] < 5.0 or v50[i] <= 0:
            continue

        # 1) 와인스타인 150일선 돌파
        if ma150[i - 1] == ma150[i - 1] and c[i - 1] <= ma150[i - 1] and c[i] > ma150[i]:
            if v[i] >= 1.5 * v50[i]:
                r = sim(o, h, l, c, i + 1)
                if r is not None:
                    R["wein"].append(r * 100)
            r2 = sim(o, h, l, c, i + 1)
            if r2 is not None:
                R["wein_up"].append(r2 * 100)

        # 2) 미너비니 추세 템플릿
        ok = (ma200[i] == ma200[i] and ma150[i] == ma150[i]
              and c[i] > ma150[i] and c[i] > ma200[i]
              and ma150[i] > ma200[i]
              and ma200[i] > ma200[i - 20]
              and lo250[i] > 0 and c[i] >= lo250[i] * 1.25
              and hi250[i] > 0 and c[i] >= hi250[i] * 0.75)
        if ok and not miner_prev:
            r = sim(o, h, l, c, i + 1)
            if r is not None:
                R["miner"].append(r * 100)
        miner_prev = ok

        # 3) VCP: 60일을 20일 3구간으로 나눠 변동폭 축소 + 20일 신고가 돌파
        if i >= 60 and hi20[i - 1] == hi20[i - 1]:
            rng = []
            for k in (2, 1, 0):
                s = i - 20 * (k + 1)
                seg_h = h[s:i - 20 * k].max(); seg_l = l[s:i - 20 * k].min()
                rng.append((seg_h - seg_l) / c[i] * 100)
            if rng[0] > rng[1] > rng[2] and c[i] > hi20[i - 1] and v[i] >= 1.5 * v50[i]:
                r = sim(o, h, l, c, i + 1)
                if r is not None:
                    R["vcp"].append(r * 100)

        # 4) 오닐 돌파 + 거래량 구간
        if hi20[i - 1] == hi20[i - 1] and c[i] > hi20[i - 1]:
            chg = (v[i] / v50[i] - 1) * 100
            r = sim(o, h, l, c, i + 1)
            if r is not None:
                VOL.append(dict(net=r * 100, chg=chg))

print(f"\n{'방식':<34}{'건수':>7}{'승률':>7}{'건당%':>8}{'합계%':>9}")
for key, label in (("wein", "1) 150일선 돌파 + 거래량 1.5배"),
                   ("wein_up", "1b) 150일선 돌파 (거래량 무관)"),
                   ("miner", "2) 미너비니 추세 템플릿"),
                   ("vcp", "3) VCP 축소 + 신고가 돌파")):
    x = np.array(R[key])
    if not len(x):
        continue
    print(f"{label:<34}{len(x):>7,}{(x>0).mean()*100:>7.1f}%{x.mean():>+8.3f}{x.sum():>+9.0f}")

d = pd.DataFrame(VOL)
print(f"\n4) 20일 신고가 돌파 · 거래량 증가율 구간별   건수   승률    건당%   합계%")
for lo, hi, lab in ((-999, 50, "50% 미만"), (50, 100, "50~100%"), (100, 150, "100~150%"), (150, 9999, "150% 이상")):
    x = d[(d.chg >= lo) & (d.chg < hi)].net.values
    if not len(x):
        continue
    print(f"   {lab:<16}{len(x):>8,}{(x>0).mean()*100:>7.1f}%{x.mean():>+8.3f}{x.sum():>+9.0f}")
x = d.net.values
print(f"   {'전체':<16}{len(x):>8,}{(x>0).mean()*100:>7.1f}%{x.mean():>+8.3f}{x.sum():>+9.0f}")
