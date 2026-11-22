"""간격 조합별 7슬롯 운용 성과 (2년)

매수 1.5% 고정, 매도 1.5 / 2.0 / 2.5% 비교. 자본 회전까지 반영.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
COND = ROOT / "data" / "low_cond.json"
FEE = 0.00015
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD, SLOTS = 3, 7
BG = 1.5
MIN_AMT, MAX_AMT, POS_MIN, SURGE = 10.0, 37.51, 89.47, 20.0
MONTHS = 24
SGS = [1.5, 2.0, 2.5, 3.0]

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


def trade(o, c, h, l, e, px, sg):
    lv = [px * (1 - BG / 100) ** k for k in range(3)]
    tg = [x * (1 + sg / 100) for x in lv]
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
            if h[j] >= tg[k]:
                net += AMTS[k] * 10_000 * (sg / 100 - FEE)
                del held[k]
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


codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
mk = pd.read_csv(ROOT / "data" / "ks11_ohlc.csv", parse_dates=["Date"]).set_index("Date")
cal = [int(d.strftime("%Y%m%d")) for d in mk.index]
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
calib = sorted({d for d in cal if d >= int(start.strftime("%Y%m%d"))})
sigs = defaultdict(list)     # sig day → list of (code, e index, px, score, pack)

packs = []
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
    pid = len(packs); packs.append((o, c, h, l, [int(x.strftime("%Y%m%d")) for x in df.index]))
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
        sigs[int(df.index[e].strftime("%Y%m%d"))].append(
            (code, e, px, cell_score(a, pos, atr[i]), pid))

print(f"[INFO] 신호일 {len(sigs):,}일")
print(f"\n{'매도':>6}{'체결':>8}{'승률':>7}{'건당%':>8}{'합계%':>9}{'평균보유':>8}"
      f"{'7슬롯 월평균':>12}{'월수익률':>9}")
for sg in SGS:
    # 7슬롯 순차 시뮬레이션
    open_pos = []
    taken = []
    for d in calib:
        open_pos = [p for p in open_pos if p[0] > d]
        busy = {p[1] for p in open_pos}
        free = SLOTS - len(open_pos)
        for code, e, px, sc, pid in sorted(sigs.get(d, []), key=lambda z: -z[3]):
            if free <= 0:
                break
            if code in busy:
                continue
            o, c, h, l, ds = packs[pid]
            net, jx = trade(o, c, h, l, e, px, sg)
            taken.append((d, ds[min(jx, len(ds) - 1)], net))
            open_pos.append((taken[-1][1], code))
            busy.add(code)
            free -= 1
    a = np.array([t[2] for t in taken]) / SLOT * 100
    cap = defaultdict(list)
    cnt = defaultdict(int)
    for d0, d1, _ in taken:
        for d in calib:
            if d0 <= d < max(d1, d0 + 1):
                cnt[d] += 1
    for d in calib:
        cap[d // 100].append(cnt[d] * SLOT)
    month_ret = []
    for m, caps in cap.items():
        avg = float(np.mean(caps)) if caps else 0
        pnl = sum(n for d0, d1, n in taken if d0 // 100 == m)
        if avg > 0:
            month_ret.append(pnl / avg * 100)
    days = [len([x for x in calib if t[0] <= x < max(t[1], t[0] + 1)]) for t in taken]
    print(f"{sg:>5.1f}%{len(a):>8,}{(a>0).mean()*100:>7.1f}%{a.mean():>+8.3f}"
          f"{a.sum():>+9.0f}{np.mean(days):>8.2f}{len(taken)/len(calib)*30:>12.1f}"
          f"{np.mean(month_ret):>+8.2f}%")
