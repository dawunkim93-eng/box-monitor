"""저변동 '조건 점수' — 유동성·밴드위치·ATR 조합으로 기대값 표준화

1) 2년 거래를 조건 칸(cell)으로 나눈다
   유동성 3구간 × 밴드위치 3구간 × ATR 3구간 = 27칸
2) 칸별 평균 손익을 표본 축소(shrinkage)해 기대값으로 쓴다
3) 기대값 백분위를 0~100 점수로 표준화
4) 전기 12개월로 칸을 만들고 후기 12개월에 적용해 예측력을 검증
산출: data/low_cond.json (칸 표 + 검증 결과)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
OUT = ROOT / "data" / "low_cond.json"

GAP, FEE = 0.015, 0.00015
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_AMT, MAX_AMT = 10.0, 37.51
PREF_TAIL = "579K"
SURGE_MAX = 20.0
K = 20.0

AMT_B = [(10, 15), (15, 25), (25, 37.52)]
POS_B = [(89.47, 94), (94, 97), (97, 100.01)]
ATR_B = [(0, 3.5), (3.5, 5.0), (5.0, 99)]


def blocked():
    m = pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"]).set_index("Date")
    dn = [int(d.strftime("%Y%m%d")) for d in m.index]
    ret = m["Close"].pct_change() * 100
    return {dn[i + 1] for i in range(1, len(dn) - 1) if ret.iloc[i] <= -3.0}


def trade(c, h, l, o, i):
    lim = float(c[i])
    if float(l[i + 1]) > lim:
        return None
    lv = [lim * (1 - GAP) ** k for k in range(3)]
    held = {0: min(float(o[i + 1]), lim)}
    net = 0.0
    for dd in range(HOLD):
        j = i + 1 + dd
        if j >= len(c):
            break
        for k in (1, 2):
            if k not in held and float(l[j]) <= lv[k]:
                held[k] = lv[k]
        for k in list(held):
            if float(h[j]) >= held[k] * (1 + GAP):
                net += AMTS[k] * 10_000 * (GAP - FEE)
                del held[k]
        if held and float(c[j]) < lv[2]:        # 매일 3차 이탈 청산
            for k in list(held):
                net += AMTS[k] * 10_000 * (float(c[j]) / held[k] - 1 - FEE)
                del held[k]
            break
        if not held:
            break
    if held:
        j = min(i + HOLD, len(c) - 1)
        for k in list(held):
            net += AMTS[k] * 10_000 * (float(c[j]) / held[k] - 1 - FEE)
    return net / SLOT * 100


def cell_of(amt, pos, atr):
    ai = next((i for i, (lo, hi) in enumerate(AMT_B) if lo <= amt < hi), None)
    pi = next((i for i, (lo, hi) in enumerate(POS_B) if lo <= pos < hi), None)
    ti = next((i for i, (lo, hi) in enumerate(ATR_B) if lo <= atr < hi), None)
    return None if None in (ai, pi, ti) else (ai, pi, ti)


def main():
    codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
    blk = blocked()
    now = pd.Timestamp.now().normalize()
    start, mid = now - pd.DateOffset(months=24), now - pd.DateOffset(months=12)

    recs = []
    for code in codes:
        if code[-1] in PREF_TAIL:
            continue
        try:
            df = pd.read_pickle(PIT / f"{code}.pkl")
        except Exception:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df["Volume"] > 0]
        if len(df) < 300:
            continue
        c = df["Close"].values.astype(float); h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float); o = df["Open"].values.astype(float)
        v = df["Volume"].values.astype(float)
        n = len(df)
        tr = np.concatenate([[h[0] - l[0]],
                             np.maximum(h[1:] - l[1:],
                                        np.maximum(np.abs(h[1:] - c[:-1]),
                                                   np.abs(l[1:] - c[:-1])))])
        atr = pd.Series(tr).ewm(alpha=1 / 14, min_periods=14).mean().values / np.maximum(c, 1e-9) * 100
        cs = pd.Series(c)
        lo30 = cs.rolling(30).min().values; hi30 = cs.rolling(30).max().values
        amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
        for i in range(260, n - 1):
            d0 = df.index[i]
            if d0 < start or lo30[i] <= 0 or hi30[i] <= lo30[i]:
                continue
            a, t = amt20[i], atr[i]
            pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
            if not (MIN_AMT <= a <= MAX_AMT) or pos < 89.47 or np.isnan(t):
                continue
            if i >= 3 and (c[i] / c[i - 3] - 1) * 100 > SURGE_MAX:   # 급등 제외
                continue
            if int(df.index[i + 1].strftime("%Y%m%d")) in blk:
                continue
            cell = cell_of(a, pos, t)
            if cell is None:
                continue
            r = trade(c, h, l, o, i)
            if r is None:
                continue
            recs.append((d0 < mid, cell, r))

    isr = [(c_, r) for is_, c_, r in recs if is_]
    oos = [(c_, r) for is_, c_, r in recs if not is_]
    avg0 = float(np.mean([r for _, r in isr]))
    win0 = float(np.mean([1 if r > 0 else 0 for _, r in isr]))
    print(f"[INFO] 전기 {len(isr):,}건(평균 {avg0:+.3f}%) · 후기 {len(oos):,}건")

    # 칸별 축소 기대값 (전기)
    agg = defaultdict(lambda: [0, 0.0, 0])
    for cell, r in isr:
        e = agg[cell]
        e[0] += 1
        e[1] += r
        e[2] += 1 if r > 0 else 0
    cells = {}
    for cell, (n, s, w) in agg.items():
        sa = (s + avg0 * K) / (n + K)
        sw = (w + win0 * K) / (n + K)
        cells[cell] = dict(cell=cell, n=n, avg=round(s / n, 3), win=round(w / n * 100, 1),
                           sa=sa, sw=sw)
    vals = sorted(c["sa"] for c in cells.values())

    def pct(x):
        lo, hi = 0, len(vals)
        while lo < hi:
            m = (lo + hi) // 2
            if vals[m] < x:
                lo = m + 1
            else:
                hi = m
        return lo / max(len(vals) - 1, 1) * 100

    for c in cells.values():
        c["score"] = round(pct(c["sa"]), 1)

    # 후기 적용 → 점수 구간별 성과
    buckets = defaultdict(lambda: [0, 0.0, 0])
    for cell, r in oos:
        c = cells.get(cell)
        if not c:
            continue
        g = ("80+" if c["score"] >= 80 else "60~80" if c["score"] >= 60
             else "40~60" if c["score"] >= 40 else "20~40" if c["score"] >= 20 else "~20")
        b = buckets[g]
        b[0] += 1
        b[1] += r
        b[2] += 1 if r > 0 else 0

    order = ["80+", "60~80", "40~60", "20~40", "~20"]
    print(f"\n{'점수':<8}{'후기건수':>10}{'승률':>8}{'평균':>9}{'합계%':>10}")
    val = []
    for g in order:
        b = buckets.get(g)
        if not b:
            continue
        print(f"{g:<8}{b[0]:>10,}{b[2]/b[0]*100:>7.1f}%{b[1]/b[0]:>+9.3f}{b[1]:>+10.1f}")
        val.append(dict(grade=g, n=b[0], win=round(b[2] / b[0] * 100, 1),
                        avg=round(b[1] / b[0], 3), total=round(b[1], 1)))

    table = []
    for cell, c in sorted(cells.items(), key=lambda x: -x[1]["score"]):
        ai, pi, ti = cell
        table.append(dict(
            amt=f"{AMT_B[ai][0]:.0f}~{AMT_B[ai][1]:.0f}억",
            pos=f"{POS_B[pi][0]:.0f}~{min(POS_B[pi][1],100):.0f}%",
            atr=f"{ATR_B[ti][0]:.1f}~{min(ATR_B[ti][1],20):.0f}%",
            n=c["n"], win=c["win"], avg=c["avg"], score=c["score"]))

    out = dict(updated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
               base=dict(avg=round(avg0, 3), win=round(win0 * 100, 1),
                         is_n=len(isr), oos_n=len(oos)),
               bands=dict(amt=[list(map(float, b)) for b in AMT_B],
                          pos=[list(map(float, b)) for b in POS_B],
                          atr=[list(map(float, b)) for b in ATR_B]),
               k=K, cells=table, validation=val,
               note="점수 = 전기 12개월 칸별 축소 기대값의 백분위. 후기 12개월로 검증.")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[INFO] 칸 {len(table)}개 → {OUT.name}")
    for t in table[:5]:
        print(f"   {t['score']:>5.1f}  {t['amt']:<12}{t['pos']:<8}{t['atr']:<10}"
              f"n={t['n']:>4} 승률 {t['win']:>4.1f}% 평균 {t['avg']:+.3f}%")


if __name__ == "__main__":
    main()
