"""저변동 7슬롯 제한 월별 성과 (종가 매수, 최근 2년)

슬롯 7개 × 21만원 = 147만원. 매일 점수 높은 순으로 빈 슬롯을 채움.
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
HOLD, SLOTS = 3, 7
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
    return {dn[i] for i in range(1, len(dn)) if ret.iloc[i] <= -3.0}, m.index


def trade(c, h, l, i):
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
            return net, int(j)
    if held:
        j = min(i + HOLD, len(c) - 1)
        for k in list(held):
            net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
        return net, int(j)
    return net, i + HOLD


codes = sorted(set(json.loads(UNIV.read_text(encoding="utf-8"))["codes"]))
blk, cal = cooldown()
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
calib = sorted({int(x.strftime("%Y%m%d")) for x in cal if x >= start})
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
        net, jx = trade(c, h, l, i)
        byday[d0].append(dict(code=code, net=net,
                              exit=int(df.index[min(jx, n - 1)].strftime("%Y%m%d")),
                              score=cell_score(a, pos, atr[i])))

# 7슬롯 순차 시뮬레이션
open_pos = []          # (exit_day, code)
taken, skipped = [], 0
for d in calib:
    open_pos = [p for p in open_pos if p[0] > d]
    busy = {p[1] for p in open_pos}
    free = SLOTS - len(open_pos)
    cand = sorted([x for x in byday.get(d, []) if x["code"] not in busy],
                  key=lambda z: -z["score"])
    for x in cand:
        if free <= 0:
            skipped += 1
            continue
        taken.append(dict(sig=d, exit=x["exit"], net=x["net"], pct=x["net"] / SLOT * 100))
        open_pos.append((x["exit"], x["code"]))
        busy.add(x["code"])            # 같은 날 중복 신호 차단
        free -= 1

import json as _j
(ROOT/'data'/'low_month7_taken.json').write_text(_j.dumps(taken), encoding='utf-8')
month = defaultdict(lambda: dict(n=0, win=0, net=0.0, cap=[]))
cnt = defaultdict(int)
for t in taken:
    _end = t["exit"] if t["exit"] > t["sig"] else t["sig"] + 1
    for d in calib:
        if t["sig"] <= d < _end:          # 청산일 종가에 슬롯이 비므로 < 로 셈
            cnt[d] += 1
_mx = max(cnt.values()) if cnt else 0
print(f"[DEBUG] 동시보유 최대 {_mx} · 일수 {len(calib)} · 평균 {sum(cnt.values())/max(len(calib),1):.2f}")
for d in calib:
    month[d // 100]["cap"].append(cnt[d] * SLOT)
for t in taken:
    m = t["sig"] // 100
    month[m]["n"] += 1
    month[m]["win"] += 1 if t["net"] > 0 else 0
    month[m]["net"] += t["net"]

rows = []
for m in sorted(month):
    x = month[m]
    if x["n"] == 0:
        continue
    cap = float(np.mean(x["cap"])) if x["cap"] else 0
    rows.append(dict(month=f"{m//100}-{m%100:02d}", n=x["n"],
                     win=round(x["win"] / x["n"] * 100, 1),
                     avg=round(x["net"] / x["n"] / SLOT * 100, 3),
                     net=round(x["net"]), cap=round(cap),
                     ret=round(x["net"] / cap * 100, 2) if cap else 0.0))

out = dict(months=rows, slots=SLOTS, slot_amt=SLOT,
           total=dict(n=len(taken), net=round(sum(r["net"] for r in rows)),
                      plus=sum(1 for r in rows if r["net"] > 0), months=len(rows),
                      skipped=skipped, signals=sum(len(v) for v in byday.values())))
(ROOT / "data" / "low_month7.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8")

print(f"[INFO] 신호 {out['total']['signals']:,}건 · 체결 {len(taken):,}건 "
      f"(제외 {skipped:,}) · 슬롯 {SLOTS}개")
print(f"\n{'월':<9}{'건수':>6}{'승률':>7}{'건당':>8}{'합계(원)':>12}{'평균자본':>11}{'수익률':>8}")
for r in rows:
    print(f"{r['month']:<9}{r['n']:>6,}{r['win']:>6.1f}%{r['avg']:>+8.3f}{r['net']:>+12,}{r['cap']:>11,}{r['ret']:>+7.2f}%")
print(f"\n[INFO] {len(rows)}개월 · 합계 {out['total']['net']:+,}원 · 플러스 "
      f"{out['total']['plus']}/{len(rows)}개월")
