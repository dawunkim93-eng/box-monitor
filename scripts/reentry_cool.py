"""최근 매수 재진입 금지 규칙 검증

같은 종목을 청산 후 며칠 쉬었다가 다시 사는 게 나은가.
A) 제한 없음 · B) 1거래일 · C) 3거래일(현행) · D) 5거래일
"""
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


def trade(o, c, h, l, i):
    e = i + 1
    if e >= len(c):
        return None, None
    prev = float(c[i])
    if float(o[e]) < prev:
        px = float(o[e])
    elif float(l[e]) <= prev:
        px = prev
    else:
        return None, None
    lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
    stop_px = px * (1 - AUTO_STOP)
    limit_entry = abs(px - float(o[e])) > 1e-9
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
            _hit = (h[j] >= held[k] * (1 + SELL_GAP)) if (dd > 0 or not limit_entry) \
                else (c[j] >= held[k] * (1 + SELL_GAP))
            if _hit:
                net += AMTS[k] * 10_000 * (SELL_GAP - FEE)
                del held[k]
        if held and l[j] <= stop_px:
            fill = min(stop_px, float(o[j])) if float(o[j]) < stop_px else stop_px
            for k in list(held):
                net += AMTS[k] * 10_000 * (fill / held[k] - 1 - FEE)
            return net, j
        if held and c[j] < lv[2]:
            for k in list(held):
                net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
            return net, j
    if held:
        j = min(e + HOLD - 1, len(c) - 1)
        for k in list(held):
            net += AMTS[k] * 10_000 * (c[j] / held[k] - 1 - FEE)
        return net, j
    return net, e


codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
sig_rows = []
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
    for i in range(270, n - 1):
        if df.index[i] < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        if not (MIN_AMT <= amt20[i] <= MAX_AMT):
            continue
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if pos < POS_MIN:
            continue
        if (c[i] / c[i - 3] - 1) * 100 > SURGE or (c[i] / c[i - 1] - 1) * 100 > SURGE1:
            continue
        net, jx = trade(o, c, h, l, i)
        if net is None:
            continue
        sig_rows.append(dict(code=code, i=i, days=days, sig=days[i],
                             exit=days[min(jx, n - 1)], net=net))

print(f"[INFO] 기본 신호 {len(sig_rows):,}건")

print(f"\n{'재진입 규칙':<22}{'건수':>7}{'승률':>7}{'건당%':>8}{'합계(원)':>13}")
for cool, label in ((0, "제한 없음"), (1, "1거래일 금지"), (3, "3거래일 금지 (현행)"), (5, "5거래일 금지")):
    last = {}
    kept = []
    for r in sorted(sig_rows, key=lambda x: (x["sig"], x["code"])):
        prev = last.get(r["code"])
        if prev is not None and cool > 0:
            # 신호일 기준 거래일 간격
            gap = sum(1 for d in r["days"] if prev < d <= r["sig"])
            if gap <= cool:
                continue
        last[r["code"]] = r["sig"]
        kept.append(r)
    net = np.array([k["net"] for k in kept])
    pct = net / SLOT * 100
    print(f"{label:<22}{len(kept):>7,}{(net>0).mean()*100:>6.1f}%{pct.mean():>+8.3f}{net.sum():>+13,.0f}")

# 7슬롯 시뮬레이션 (신호 순서대로 채움, 청산일에 자리 반환)
mk = pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"]).set_index("Date")
cal = sorted({int(d.strftime("%Y%m%d")) for d in mk.index
              if d >= start})
print(f"\n[7슬롯] {'재진입 규칙':<18}{'체결':>7}{'승률':>7}{'건당%':>8}{'합계(원)':>13}{'최대자본':>12}")
for cool, label in ((0, "제한 없음"), (1, "1거래일"), (3, "3거래일 (현행)")):
    last, kept = {}, []
    for r in sorted(sig_rows, key=lambda x: (x["sig"], x["code"])):
        prev = last.get(r["code"])
        if prev is not None and cool > 0:
            gap = sum(1 for d in r["days"] if prev < d <= r["sig"])
            if gap <= cool:
                continue
        last[r["code"]] = r["sig"]
        kept.append(r)
    by_sig = {}
    for r in kept:
        by_sig.setdefault(r["sig"], []).append(r)
    slots = []          # (청산일, 종목, net)
    taken = []
    for d in cal:
        slots = [s for s in slots if s[0] > d]        # 청산일 지나면 자리 반환
        for r in by_sig.get(d, []):
            if len(slots) < 7:
                slots.append((r["exit"], r["code"], r["net"]))
                taken.append(r["net"])
    t = np.array(taken)
    print(f"{'':<26}{label:<18}{len(t):>7,}{(t>0).mean()*100:>6.1f}%{t.mean()/SLOT*100:>+8.3f}{t.sum():>+13,.0f}{7*SLOT:>12,}")
