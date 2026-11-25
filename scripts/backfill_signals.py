"""누락된 신호를 원장에 채워 넣는다 (11/13~)

매일 15:18 실행 때 저장하지 못한 '조건 충족 전체' 목록을 PIT 데이터로 재계산해
원장에 기록한다. 보유중·최근 매수·급등은 사유를 붙여 남긴다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
PICKS = ROOT / "data" / "paper_picks.json"
COND = ROOT / "data" / "low_cond.json"
MIN_AMT, MAX_AMT, POS_MIN, ATR_MAX = 10.0, 37.51, 89.47, 5.85
SURGE, SURGE1 = 40.0, 20.0
HOLD = 3
DATES = [20261120, 20261121, 20261124]


def score(amt, pos, atr):
    try:
        c = json.loads(COND.read_text(encoding="utf-8"))
        bands = c["bands"]
    except Exception:
        return None

    def idx(v, bs):
        for i, (lo, hi) in enumerate(bs):
            if lo <= v < hi:
                return i
        return None
    ai, pi, ti = idx(amt, bands["amt"]), idx(pos, bands["pos"]), idx(atr, bands["atr"])
    if None in (ai, pi, ti):
        return None
    alo, ahi = bands["amt"][ai]
    plo, phi = bands["pos"][pi]
    tlo, thi = bands["atr"][ti]
    key = (f"{alo:.0f}~{ahi:.0f}억", f"{plo:.0f}~{min(phi, 100):.0f}%",
           f"{tlo:.1f}~{min(thi, 20):.0f}%")
    for cell in c.get("cells", []):
        if (cell.get("amt") == key[0] and cell.get("pos") == key[1]
                and cell.get("atr") == key[2]):
            return cell.get("score")
    return None


picks = json.loads(PICKS.read_text(encoding="utf-8"))
have = {(p["code"], p["sig"]) for p in picks}
cal = sorted({int(x.strftime("%Y%m%d")) for x in
              pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"])["Date"]})
names = {x["code"]: x["name"] for x in
         json.loads(UNIV.read_text(encoding="utf-8"))["items"]}
codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]


def held_or_recent(code, day, existing):
    """day 시점에 보유중이거나 최근 3거래일 내 매수였는지"""
    prev = [p for p in existing if p["code"] == code and p.get("sig", 0) < day]
    if not prev:
        return None
    last = max(prev, key=lambda p: p["sig"])
    gap = len([d for d in cal if last["sig"] < d <= day])
    if gap <= HOLD:
        return "보유중" if gap <= HOLD else "최근 매수"
    return None


added = []
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
    tr = np.concatenate([[h[0] - l[0]],
                         np.maximum(h[1:] - l[1:],
                                    np.maximum(np.abs(h[1:] - c[:-1]),
                                               np.abs(l[1:] - c[:-1])))])
    atr = pd.Series(tr).ewm(alpha=1 / 14, min_periods=14).mean().values / np.maximum(c, 1e-9)
    cs = pd.Series(c)
    lo30 = cs.rolling(30).min().values; hi30 = cs.rolling(30).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    days = [int(x.strftime("%Y%m%d")) for x in df.index]
    for D in DATES:
        if D not in days:
            continue
        i = days.index(D)
        if i < 3 or lo30[i] <= 0 or hi30[i] <= lo30[i]:
            continue
        amt, pos, a = amt20[i], (c[i] - lo30[i]) / (hi30[i] - lo30[i]) * 100, atr[i] * 100
        if not (MIN_AMT <= amt <= MAX_AMT) or pos < POS_MIN:
            continue
        if (code, D) in have:
            continue
        s1 = (c[i] / c[i - 1] - 1) * 100 > SURGE1
        s3 = (c[i] / c[i - 3] - 1) * 100 > SURGE
        skip = ("급등 제외" if (s1 or s3) else held_or_recent(code, D, picks))
        picks.append(dict(code=code, name=names.get(code, code), sig=D,
                          limit=round(float(c[i])), cond="A", held=False,
                          amt=round(float(amt), 1), pos=round(float(pos), 1),
                          atr=round(float(a), 2), open=None, gap=None,
                          score=score(amt, pos, a),
                          skip=skip, backfilled=True))
        have.add((code, D))
        added.append((D, names.get(code, code), skip))

picks.sort(key=lambda p: (p["sig"], p["code"]))
PICKS.write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[INFO] 백필 {len(added)}건 추가 → 원장 {len(picks)}건")
from collections import Counter
for D in DATES:
    xs = [x for x in added if x[0] == D]
    if xs:
        print(f"  {D}: {len(xs)}건 (제외 {sum(1 for x in xs if x[2])}건)")
