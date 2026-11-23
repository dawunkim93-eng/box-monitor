"""11/20 후보 소급 등록 (당시 쿨다운으로 누락된 날)

11/20 종가 기준 조건 충족 종목을 원장에 넣는다.
보유 중이던 종목은 제외.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
sys.path.insert(0, str(ROOT / "scripts"))
from daily_pick import load, naver_close, evaluate, MIN_AMT, SURGE_MAX, PREF_TAIL  # noqa: E402

PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
PICKS = ROOT / "data" / "paper_picks.json"
COND = ROOT / "data" / "low_cond.json"
MAX_AMT, POS_MIN = 37.51, 89.47
SIG = 20261120

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


_u = json.loads(UNIV.read_text(encoding="utf-8"))
NAMES = {x["code"]: x["name"] for x in _u.get("items", [])}
codes = sorted(set(_u["codes"]))

picks = json.loads(PICKS.read_text(encoding="utf-8")) if PICKS.exists() else []
if any(p.get("sig") == SIG for p in picks):
    print("[INFO] 이미 11/20 기록 있음")
    sys.exit()

# 11/20에 보유 중이던 종목
held = set()
for p in picks:
    d = load(p["code"])
    if d is None:
        continue
    r = evaluate(p["code"], d, p["sig"], float(p["limit"]))
    if not r:
        continue
    ent = r.get("entry")
    ex = r.get("exit")
    if ent and ent <= SIG and (ex is None or ex >= SIG):
        held.add(p["code"])

added = []
for code in codes:
    if code[-1] in PREF_TAIL or code in held:
        continue
    df = load(code)
    if df is None or len(df) < 300:
        continue
    days = [int(x.strftime("%Y%m%d")) for x in df.index]
    if SIG not in days:
        continue
    i = days.index(SIG)
    if i < 30 or i >= len(df) - 1:
        continue
    c = df["Close"].values.astype(float); h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float); v = df["Volume"].values.astype(float)
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = pd.Series(np.concatenate([[h[0] - l[0]], tr])).ewm(alpha=1/14, min_periods=14) \
        .mean().values / np.maximum(c, 1e-9) * 100
    cs = pd.Series(c)
    lo30 = cs.rolling(30).min().values[i]; hi30 = cs.rolling(30).max().values[i]
    if lo30 <= 0 or hi30 <= lo30:
        continue
    amt = pd.Series(c * v).rolling(20).mean().values[i] / 1e8
    pos = (c[i] - lo30) / (hi30 - lo30) * 100
    if not (MIN_AMT <= amt <= MAX_AMT) or pos < POS_MIN or np.isnan(atr[i]):
        continue
    if (c[i] / c[i - 3] - 1) * 100 > SURGE_MAX:
        continue
    lim = naver_close(code, SIG) or float(c[i])
    added.append(dict(code=code, name=NAMES.get(code, code), sig=SIG,
                      limit=round(lim), cond="소급",
                      amt=round(amt, 1), pos=round(pos, 1), atr=round(atr[i], 2),
                      open=round(float(df["Open"].values[i])),
                      gap=0.0, score=cell_score(amt, pos, atr[i]),
                      backfill=True))

added.sort(key=lambda x: -(x["score"] or 0))
picks.extend(added)
PICKS.write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[INFO] 11/20 소급 {len(added)}건 (보유중 제외 {len(held)})")
for x in added:
    print(f"  {x['score']:>5.1f} {x['name']:<14}{x['limit']:>9,}")
