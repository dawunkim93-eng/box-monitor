"""저변동 전략 종목별 점수 표준화 → data/low_score.json

방법:
  1) 2년 백테스트(저변동 규칙)로 종목별 신호·승률·평균손익을 모은다
  2) 표본이 적은 종목은 전체 평균 쪽으로 축소(shrinkage)해 과대평가를 막는다
     shrunk = (관측값×n + 전체평균×k) / (n + k)
  3) 축소된 기대값(60%)과 승률(40%)의 백분위를 합쳐 0~100 점수로 표준화
  4) 신호 10건 미만은 점수를 주지 않는다(표본 부족)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
OUT = ROOT / "data" / "low_score.json"

BUY_GAP, SELL_GAP, FEE = 0.015, 0.020, 0.00015
AUTO_STOP = 0.035
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MONTHS = 24
MIN_N = 10
K = 8.0
MIN_AMT, MAX_AMT = 10.0, 37.51
PREF_TAIL = "579K"


def blocked():
    m = pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"]).set_index("Date")
    dn = [int(d.strftime("%Y%m%d")) for d in m.index]
    ret = m["Close"].pct_change() * 100
    return {dn[i + 1] for i in range(1, len(dn) - 1) if ret.iloc[i] <= -3.0}


def trade(c, h, l, o, i):
    lim = float(c[i])
    # 종가 매수 — 항상 체결
    lv = [lim * (1 - BUY_GAP) ** k for k in range(3)]
    stop_px = lim * (1 - AUTO_STOP)
    held = {0: lim}
    net = 0.0
    for dd in range(HOLD):
        j = i + 1 + dd
        if j >= len(c):
            break
        for k in (1, 2):
            if k not in held and float(l[j]) <= lv[k]:
                held[k] = lv[k]
        for k in list(held):
            _hit = (float(h[j]) >= held[k] * (1 + SELL_GAP)) if (dd > 0 or not limit_entry) \
                else (float(c[j]) >= held[k] * (1 + SELL_GAP))
            if _hit:
                net += AMTS[k] * 10_000 * (SELL_GAP - FEE)
                del held[k]
        if held and l[j] <= stop_px:
            _fill = min(stop_px, float(o[j])) if float(o[j]) < stop_px else stop_px
            for k in list(held):
                net += AMTS[k] * 10_000 * (_fill / held[k] - 1 - FEE)
                del held[k]
            return net, int(j)
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


def main():
    codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
    master = pd.read_pickle(PIT / "master.pkl")
    names = dict(zip(master["code"].astype(str), master["name"].astype(str)))
    blk = blocked()
    start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)

    per = defaultdict(list)
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
        c = df["Close"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        o = df["Open"].values.astype(float)
        v = df["Volume"].values.astype(float)
        n = len(df)
        cs = pd.Series(c)
        lo30 = cs.rolling(30).min().values
        hi30 = cs.rolling(30).max().values
        amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
        for i in range(260, n - 1):
            if df.index[i] < start:
                continue
            if lo30[i] <= 0 or hi30[i] <= lo30[i]:
                continue
            a = amt20[i]
            pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
            if not (MIN_AMT <= a <= MAX_AMT) or pos < 89.47:
                continue
            ed = int(df.index[i].strftime("%Y%m%d"))
            if ed in blk:
                continue
            r = trade(c, h, l, o, i)
            if r is None:
                continue
            per[code].append(r)

    allr = [x for v in per.values() for x in v]
    win0 = float(np.mean([1 if x > 0 else 0 for x in allr]))
    avg0 = float(np.mean(allr))
    print(f"[INFO] 전체 {len(allr):,}건 · 승률 {win0*100:.1f}% · 평균 {avg0:+.3f}% · "
          f"종목 {len(per)}")

    rows = []
    for code, v in per.items():
        n = len(v)
        if n < MIN_N:
            continue
        w = float(np.mean([1 if x > 0 else 0 for x in v]))
        a = float(np.mean(v))
        sw = (w * n + win0 * K) / (n + K)
        sa = (a * n + avg0 * K) / (n + K)
        rows.append(dict(code=code, name=names.get(code, code), n=n,
                         win=round(w * 100, 1), avg=round(a, 3),
                         total=round(sum(v), 1), worst=round(min(v), 2),
                         sw=sw, sa=sa))

    sa_sorted = sorted(r["sa"] for r in rows)
    sw_sorted = sorted(r["sw"] for r in rows)

    def pct(arr, x):
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo / max(len(arr) - 1, 1) * 100

    for r in rows:
        r["score"] = round(0.6 * pct(sa_sorted, r["sa"]) + 0.4 * pct(sw_sorted, r["sw"]), 1)
        r["grade"] = ("S" if r["score"] >= 85 else "A" if r["score"] >= 70
                      else "B" if r["score"] >= 45 else "C" if r["score"] >= 25 else "D")
    rows.sort(key=lambda x: -x["score"])
    keep = ("code", "name", "n", "win", "avg", "total", "worst", "score", "grade")
    out = dict(
        updated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        method=dict(months=MONTHS, min_n=MIN_N, k=K,
                    base_win=round(win0 * 100, 1), base_avg=round(avg0, 3),
                    rule="0.6×기대값 백분위 + 0.4×승률 백분위 (표본 축소 적용)"),
        stocks=len(rows), trades=len(allr),
        top=[{k: r[k] for k in keep} for r in rows[:30]],
        bottom=[{k: r[k] for k in keep} for r in rows[-15:]],
    )
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[INFO] 점수 산출 {len(rows)}종목 → {OUT.name}")
    for r in rows[:8]:
        print(f"   {r['grade']} {r['score']:>5.1f}  {r['name']:<12} n={r['n']:>3} "
              f"승률 {r['win']:>4.1f}% 평균 {r['avg']:+.3f}%")
    print("   ...")
    for r in rows[-4:]:
        print(f"   {r['grade']} {r['score']:>5.1f}  {r['name']:<12} n={r['n']:>3} "
              f"승률 {r['win']:>4.1f}% 평균 {r['avg']:+.3f}%")


if __name__ == "__main__":
    main()
