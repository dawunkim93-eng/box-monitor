"""진입 시각을 바꾸면 어떻게 되나 (분봉 기준, 최근 3주)

가설: 아침(09:00)은 변동성이 커서 손절에 걸리기 쉽다.
→ 10~14시 등 변동성이 낮은 시간에 들어가면 나은가?

분봉 데이터가 최근 2~3주만 있어 표본이 작다는 점을 감안해야 한다.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
MIN = ROOT / "data" / "minute"
STOP, TGT = 0.035, 0.020
MIN_AMT, MAX_AMT, POS_MIN = 10.0, 37.51, 89.47
SURGE, SURGE1 = 40.0, 20.0
START, END = 20261106, 20261125
TIMES = ["0900", "0930", "1000", "1030", "1100", "1200", "1300", "1400", "1500"]


def load_min(code):
    p = MIN / f"{code}.csv.gz"
    if not p.exists():
        return None
    d = {}
    with gzip.open(p, "rt") as f:
        for line in f:
            t = line.strip().split(",")
            if len(t) == 3 and t[0].isdigit():
                d[t[0]] = float(t[1])
    return d or None


def price_at(day_bars, hhmm):
    """해당 시각 이하의 마지막 분봉 가격"""
    cand = [(t, p) for t, p in day_bars if t[8:12] <= hhmm]
    if not cand:
        return None
    return cand[-1][1]


codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
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
    c = df["Close"].values.astype(float); v = df["Volume"].values.astype(float)
    h = df["High"].values.astype(float); l = df["Low"].values.astype(float)
    cs = pd.Series(c)
    lo30 = cs.rolling(30).min().values; hi30 = cs.rolling(30).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    days = [int(x.strftime("%Y%m%d")) for x in df.index]
    for i in range(270, len(df) - 3):
        d0 = days[i]
        if not (START <= d0 <= END):
            continue
        if lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        if not (MIN_AMT <= amt20[i] <= MAX_AMT):
            continue
        pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
        if pos < POS_MIN:
            continue
        if (c[i] / c[i - 3] - 1) * 100 > SURGE or (c[i] / c[i - 1] - 1) * 100 > SURGE1:
            continue
        sig_rows.append(dict(code=code, i=i, days=days, h=h, l=l, c=c))

print(f"[INFO] 신호 {len(sig_rows):,}건 (분봉 구간 {START}~{END})")

out = {}
n_cov = 0
for r in sig_rows:
    code, i = r["code"], r["i"]
    buy_day = str(r["days"][i + 1])
    mb = load_min(code)
    if not mb:
        continue
    db = sorted([(t, p) for t, p in mb.items() if t[:8] == buy_day])
    if not db:
        continue
    n_cov += 1
    # 이후 거래일 일봉 (D+1, D+2)
    later = [(int(r["days"][j]), r["h"][j], r["l"][j], r["c"][j])
             for j in range(i + 2, min(i + 5, len(r["days"])))]
    for tm in TIMES:
        p0 = price_at(db, tm)
        if not p0 or p0 <= 0:
            continue
        stop_px, tgt_px = p0 * (1 - STOP), p0 * (1 + TGT)
        res = None
        # 당일: 진입 이후 분봉
        after = [(t, p) for t, p in db if t[8:12] > tm]
        for t, p in after:
            if p <= stop_px:
                res = -STOP
                break
            if p >= tgt_px:
                res = TGT
                break
        if res is None:
            for dd, hh, ll, cc in later:
                if hh >= tgt_px:
                    res = TGT
                    break
                if ll <= stop_px:
                    res = -STOP
                    break
            if res is None:
                res = (r["c"][i + 3] / p0 - 1) if i + 3 < len(r["c"]) else 0.0
        out.setdefault(tm, []).append(res)

print(f"[INFO] 분봉 확인 가능 {n_cov:,}건\n")
print(f"{'진입시각':<10}{'건수':>7}{'승률':>8}{'건당%':>9}{'합계%':>9}")
for tm in TIMES:
    x = np.array(out.get(tm, [])) * 100
    if not len(x):
        continue
    lab = f"{tm[:2]}:{tm[2:]}"
    print(f"{lab:<10}{len(x):>7,}{(x>0).mean()*100:>7.1f}%{x.mean():>+9.3f}{x.sum():>+9.0f}")

# 시간대별 변동성 (분당 절대 변동, 진입 후보 종목 기준)
vol = {tm: [] for tm in TIMES[:-1]}
for r in sig_rows[:400]:
    mb = load_min(r["code"])
    if not mb:
        continue
    bd = str(r["days"][r["i"] + 1])
    bars = sorted([(t, p) for t, p in mb.items() if t[:8] == bd])
    for k in range(1, len(bars)):
        t, p = bars[k]
        pp = bars[k - 1][1]
        if pp > 0:
            tm = t[8:10] + "00"
            if tm in vol:
                vol[tm].append(abs(p / pp - 1) * 100)
print(f"\n{'시간대':<8}{'분당 평균 변동':>14}")
for tm in TIMES[:-1]:
    a = np.array(vol.get(tm, []))
    if len(a):
        print(f"{tm[:2]}:00{'':<4}{a.mean():>13.4f}%")
