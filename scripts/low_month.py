"""저변동 전략 2년 월별 성과 (종가 매수 기준)

월별: 건수, 승률, 건당 수익률, 합계(원), 평균 투입자본, 자본 대비 수익률
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from nasdaq_rule import shift_to_close  # noqa: E402

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
BUY_GAP, SELL_GAP, FEE = 0.015, 0.020, 0.00015
AUTO_STOP = 0.035
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000          # 21만원
HOLD = 3
SLIP_BUY = float(os.environ.get("SLIP_BUY", "0")) / 100    # 매수 체결 불리(%)
SLIP_SELL = float(os.environ.get("SLIP_SELL", "0")) / 100  # 매도 체결 불리(%)
SLIP_STOP = float(os.environ.get("SLIP_STOP", "0")) / 100  # 자동손절 체결 불리(%)
MIN_AMT, MAX_AMT = 10.0, 37.51
POS_MIN = 89.47
SURGE = 40.0
PREF_TAIL = "579K"
MONTHS = 24


def cooldown_days():
    m = pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"]).set_index("Date")
    dn = [int(d.strftime("%Y%m%d")) for d in m.index]
    ret = m["Close"].pct_change() * 100
    return {dn[i] for i in range(1, len(dn)) if ret.iloc[i] <= -3.0}, m.index


def trade(o, c, h, l, i, days=None):
    """반환: (순익 원, 청산 인덱스) 또는 미체결이면 (None, None)"""
    e = i + 1
    if e >= len(c):
        return None, None
    _shift, _ = shift_to_close(days[e]) if days else (False, None)
    if _shift:                       # 나스닥 급락 다음날 → 종가 매수
        px = float(c[e])
        _st = e + 1
    else:
        prev = float(c[i])
        if float(o[e]) < prev:
            px = float(o[e])
        elif float(l[e]) <= prev:
            px = prev
        else:
            return None, None
        _st = e
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    tg = [x * (1 + SELL_GAP) for x in lv]
    stop_px = px * (1 - AUTO_STOP)
    limit_entry = abs(px - float(o[e])) > 1e-9
    held = {0: px * (1 + SLIP_BUY)}
    net = 0.0
    for dd in range(_st - e, _st - e + HOLD):
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
                net += AMTS[k] * 10_000 * (SELL_GAP - SLIP_SELL - FEE)
                del held[k]
        if held and l[j] <= stop_px:              # 장중 자동매도
            fill = min(stop_px, float(o[j])) if float(o[j]) < stop_px else stop_px
            for k in list(held):
                net += AMTS[k] * 10_000 * (fill / held[k] - 1 - FEE - SLIP_STOP)
                del held[k]
            return net, j
        if held and c[j] < lv[2]:                 # 종가 3차 이탈
            for k in list(held):
                net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE - SLIP_SELL)
                del held[k]
            return net, j
    if held:
        j = min(_st + HOLD - 1, len(c) - 1)
        for k in list(held):
            net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE - SLIP_SELL)
        return net, j
    return net, e


codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
blk, cal = cooldown_days()
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
trades = []

for code in codes:
    if code[-1] in PREF_TAIL:
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
    for i in range(270, n - 1):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        a = amt20[i]
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if not (MIN_AMT <= a <= MAX_AMT) or pos < POS_MIN:
            continue
        if (c[i] / c[i - 3] - 1) * 100 > SURGE:
            continue
        d0 = int(df.index[i].strftime("%Y%m%d"))
        if d0 in blk:
            continue
        net, jx = trade(o, c, h, l, i, days)
        if net is None:
            continue
        trades.append(dict(sig=d0, exit=int(df.index[min(jx, n - 1)].strftime("%Y%m%d")),
                           net=net, pct=net / SLOT * 100))

print(f"[INFO] 거래 {len(trades):,}건 · 기간 {trades[0]['sig']}~{trades[-1]['sig']}")

# 일별 동시 보유 수 → 자본
calib = sorted({int(x.strftime("%Y%m%d")) for x in cal if x >= start})
open_cnt = defaultdict(int)
for t in trades:
    _end = t["exit"] if t["exit"] > t["sig"] else t["sig"] + 1
    for d in calib:
        if t["sig"] <= d < _end:          # 청산일 종가에 자본이 빠지므로 < 로 셈
            open_cnt[d] += 1

month = defaultdict(lambda: dict(n=0, win=0, net=0.0, cap=[], exits=0))
for d in calib:
    m = d // 100
    month[m]["cap"].append(open_cnt[d] * SLOT)
for t in trades:
    m = t["sig"] // 100
    month[m]["n"] += 1
    month[m]["win"] += 1 if t["net"] > 0 else 0
    month[m]["net"] += t["net"]
    month[t["exit"] // 100]["exits"] += 1

rows = []
for m in sorted(month):
    x = month[m]
    if x["n"] == 0:
        continue
    cap = float(np.mean(x["cap"])) if x["cap"] else 0.0
    rows.append(dict(month=f"{m//100}-{m%100:02d}",
                     n=x["n"], win=round(x["win"] / x["n"] * 100, 1),
                     avg=round(x["net"] / x["n"] / SLOT * 100, 3),
                     net=round(x["net"]),
                     cap=round(cap),
                     ret=round(x["net"] / cap * 100, 2) if cap else 0.0))

tot_net = sum(r["net"] for r in rows)
peak = max(r["cap"] for r in rows)
out = dict(months=rows, total=dict(n=sum(r["n"] for r in rows), net=round(tot_net),
                                   cap_peak=peak,
                                   months=len(rows),
                                   plus=sum(1 for r in rows if r["net"] > 0)),
           window=dict(start=str(start.date()), slot=SLOT, hold=HOLD))
(ROOT / "data" / "low_month.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                              encoding="utf-8")

print(f"\n{'월':<9}{'건수':>6}{'승률':>7}{'건당':>8}{'합계(원)':>12}{'평균자본':>12}{'수익률':>8}")
for r in rows:
    print(f"{r['month']:<9}{r['n']:>6,}{r['win']:>6.1f}%{r['avg']:>+8.3f}{r['net']:>+12,}{r['cap']:>12,}{r['ret']:>+7.2f}%")
print(f"\n[INFO] {len(rows)}개월 · 합계 {tot_net:+,.0f}원 · 플러스 "
      f"{out['total']['plus']}/{len(rows)}개월 · 최대 자본 {peak:,}원")
