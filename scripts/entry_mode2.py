"""진입 시점 비교 — 종가 매수 vs 다음날 아침(시가/전일종가) 매수 (2년)

아침 매수 규칙:
  - 09:00 시가가 전일 종가보다 낮으면 시가에 시장가 매수
  - 시가가 전일 종가보다 높으면 전일 종가에 지정가 → 당일 저가가 닿으면 체결
  - 끝까지 안 닿으면 미체결(매수 없음)
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
GAP, FEE = 0.015, 0.00015
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_AMT, MAX_AMT, POS_MIN, SURGE = 10.0, 37.51, 89.47, 20.0
MONTHS = 24

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


def cooldown():
    m = pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"]).set_index("Date")
    dn = [int(d.strftime("%Y%m%d")) for d in m.index]
    ret = m["Close"].pct_change() * 100
    return {dn[i] for i in range(1, len(dn)) if ret.iloc[i] <= -3.0}


def run_close(c, h, l, i):
    """신호일 종가 매수 (현행)."""
    lim = c[i]
    lv = [lim * (1 - GAP) ** k for k in range(3)]
    held = {0: lim}
    net = 0.0
    for dd in range(HOLD):
        j = i + 1 + dd
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
        j = min(i + HOLD, len(c) - 1)
        for k in list(held):
            net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
    return net, None


def run_morning(o, c, h, l, i):
    """다음날 아침 매수: 시가<전일종가 → 시가 / 그 외 → 전일종가 지정가."""
    e = i + 1
    if e >= len(c):
        return None, None
    prev = c[i]
    if o[e] < prev:
        px = o[e]
    elif l[e] <= prev:
        px = prev
    else:
        return 0.0, "미체결"
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
            return net, None
    if held:
        j = min(e + HOLD - 1, len(c) - 1)
        for k in list(held):
            net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
    return net, None


codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
blk = cooldown()
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
res = {"종가매수": [], "아침매수": []}
miss = 0
byday = defaultdict(list)

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
    for i in range(270, n - 1):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        a = amt20[i]
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if not (MIN_AMT <= a <= MAX_AMT) or pos < POS_MIN or np.isnan(atr[i]):
            continue
        if (c[i] / c[i - 3] - 1) * 100 > SURGE:
            continue
        d0 = int(df.index[i].strftime("%Y%m%d"))
        if d0 in blk:
            continue
        nc, _ = run_close(c, h, l, i)
        nm, flag = run_morning(o, c, h, l, i)
        byday[d0].append(dict(close=nc, morning=nm, miss=(flag == "미체결"),
                              score=cell_score(a, pos, atr[i])))

rows = [x for v in byday.values() for x in v]
print(f"[INFO] 후보 {len(rows):,}건 · 신호일 {len(byday):,}일")

def stat(vals, label, picks=None):
    a = np.array(vals)
    print(f"{label:<12}{len(a):>8,}{(a>0).mean()*100:>7.1f}%{a.mean():>+9.3f}"
          f"{a.sum():>+9.0f}{(a<=-5).mean()*100:>6.1f}%")

print(f"\n{'방식':<12}{'건수':>8}{'승률':>7}{'건당%':>9}{'합계%':>9}{'-5%↓':>7}")
close_r = [x["close"] / SLOT * 100 for x in rows]
mor_r = [x["morning"] / SLOT * 100 for x in rows if not x["miss"]]
stat(close_r, "종가매수")
stat(mor_r, "아침매수")
print(f"미체결 {sum(1 for x in rows if x['miss']):,}건 "
      f"({sum(1 for x in rows if x['miss'])/len(rows)*100:.1f}%)")

print(f"\n[하루 3건 · 점수 상위]")
print(f"{'방식':<12}{'건수':>8}{'승률':>7}{'건당%':>9}{'합계%':>9}{'-5%↓':>7}")
p_close, p_mor = [], []
for d, lst in byday.items():
    for x in sorted(lst, key=lambda z: -z["score"])[:3]:
        p_close.append(x["close"] / SLOT * 100)
        if not x["miss"]:
            p_mor.append(x["morning"] / SLOT * 100)
stat(p_close, "종가매수")
stat(p_mor, "아침매수")
