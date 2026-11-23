"""3차 풀체결 후 손절 손실 보완안 검증 (2년, 아침 매수 · 매도 +2.0%)

1) 차수별 체결 수에 따른 성과 분해
2) 손절 기준 완화 (-2.0 / -2.5 / -3.0 / -3.5 / -4.0%)
3) 3차 금액 축소 (6:6:9 / 6:6:6 / 5:5:7 / 5:5:5)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
FEE = 0.00015
BUY_GAP, SELL_GAP = 0.015, 0.020
HOLD = 3
MIN_AMT, MAX_AMT, POS_MIN, SURGE = 10.0, 37.51, 89.47, 20.0
MONTHS = 24


def run(o, c, h, l, e, px, stop_x, amts):
    """stop_x: 1차 대비 손절 기준 (0.03 = -3%)"""
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    tg = [x * (1 + SELL_GAP) for x in lv]
    stop_line = px * (1 - stop_x)
    held = {0: 0}
    rungs = 1
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
                del held[k]
        if held and c[j] < stop_line:
            pnl = 0.0
            for k in list(held):
                pnl += amts[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
            for k in range(3):
                pass
            # 이미 익절한 차수 몫 더하기
            return pnl, "손절", rungs
        if not held:
            return 0.0, "익절", rungs
    pnl = 0.0
    j = min(e + HOLD - 1, len(c) - 1)
    for k in list(held):
        pnl += amts[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
    return pnl, "만기", rungs


# 익절 몫까지 포함한 총손익 계산이 필요하므로 별도 구현
def run2(o, c, h, l, e, px, stop_x, amts):
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    tg = [x * (1 + SELL_GAP) for x in lv]
    stop_line = px * (1 - stop_x)
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
                net += amts[k] * 10_000 * (SELL_GAP - FEE)
                del held[k]
        if held and c[j] < stop_line:
            for k in list(held):
                net += amts[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
            return net, "손절", rungs
        if not held:
            return net, "익절", rungs
    j = min(e + HOLD - 1, len(c) - 1)
    for k in list(held):
        net += amts[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
    return net, "만기", rungs


codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
packs, sigs = [], []
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
    pid = len(packs); packs.append((o, c, h, l))
    for i in range(270, n - 2):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        a = amt20[i]
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if not (MIN_AMT <= a <= MAX_AMT) or pos < POS_MIN:
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
        sigs.append((pid, e, px))

A = [6, 6, 9]
SLOT = sum(A) * 10_000
print(f"[INFO] 체결 신호 {len(sigs):,}건")

print("\n[차수 체결 수별 성과]")
print(f"{'차수':<10}{'건수':>8}{'비중':>7}{'승률':>7}{'건당%':>9}{'합계%':>10}{'손실기여':>10}")
res = [run2(*packs[pid], e, px, 0.03, A) for pid, e, px in sigs]
for r in (1, 2, 3):
    vals = np.array([x[0] for x in res if x[2] == r]) / SLOT * 100
    if not len(vals):
        print(f"{r}차까지  없음")
        continue
    print(f"{r}차까지{'':<5}{len(vals):>8,}{len(vals)/len(res)*100:>6.1f}%"
          f"{(vals>0).mean()*100:>7.1f}%{vals.mean():>+9.3f}{vals.sum():>+10.0f}"
          f"{vals[vals<0].sum() if (vals<0).any() else 0:>+10.0f}")

print("\n[손절 기준 완화]")
print(f"{'기준':<8}{'건수':>8}{'승률':>7}{'건당%':>9}{'합계%':>10}{'-5%↓':>7}{'손절건':>8}")
for sx in (0.02, 0.025, 0.03, 0.035, 0.04, 0.05):
    r2 = [run2(*packs[pid], e, px, sx, A) for pid, e, px in sigs]
    vals = np.array([x[0] for x in r2]) / SLOT * 100
    nstop = sum(1 for x in r2 if x[1] == "손절")
    print(f"-{sx*100:.1f}%{'':<3}{len(vals):>8,}{(vals>0).mean()*100:>7.1f}%"
          f"{vals.mean():>+9.3f}{vals.sum():>+10.0f}{(vals<=-5).mean()*100:>6.1f}%{nstop:>8,}")

print("\n[평균단가 기준 손절]")
def run_avg(o, c, h, l, e, px, stop_x, amts):
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
                net += amts[k] * 10_000 * (SELL_GAP - FEE)
                del held[k]
        if held:
            w = sum(amts[k] for k in held)
            avg = sum(amts[k] * lv[k] for k in held) / w
            if c[j] < avg * (1 - stop_x):
                for k in list(held):
                    net += amts[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
                return net, "손절", rungs
        if not held:
            return net, "익절", rungs
    j = min(e + HOLD - 1, len(c) - 1)
    for k in list(held):
        net += amts[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
    return net, "만기", rungs

print(f"{'기준':<10}{'건수':>8}{'승률':>7}{'건당%':>9}{'합계%':>10}{'-5%↓':>7}")
for sx in (0.01, 0.015, 0.02, 0.03):
    r4 = [run_avg(*packs[pid], e, px, sx, A) for pid, e, px in sigs]
    vals = np.array([x[0] for x in r4]) / SLOT * 100
    print(f"평균-{sx*100:.1f}%{'':<3}{len(vals):>8,}{(vals>0).mean()*100:>7.1f}%"
          f"{vals.mean():>+9.3f}{vals.sum():>+10.0f}{(vals<=-5).mean()*100:>6.1f}%")

print("\n[3차 금액 축소]")
print(f"{'배분':<12}{'슬롯':>8}{'승률':>7}{'건당%':>9}{'합계%':>10}")
for amts in ([6, 6, 9], [6, 6, 6], [5, 5, 7], [5, 5, 5], [7, 6, 8]):
    slot = sum(amts) * 10_000
    r3 = [run2(*packs[pid], e, px, 0.03, amts) for pid, e, px in sigs]
    vals = np.array([x[0] for x in r3]) / slot * 100
    lbl = ":".join(str(x) for x in amts)
    print(f"{lbl:<12}{slot:>8,}{(vals>0).mean()*100:>7.1f}%{vals.mean():>+9.3f}{vals.sum():>+10.0f}")
