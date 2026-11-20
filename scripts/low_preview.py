"""오늘 후보 미리보기 (원장에 기록하지 않음)"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/lucky/box-monitor")
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from daily_pick import load, features, MIN_AMT, SURGE_MAX, PREF_TAIL  # noqa: E402

COND = ROOT / "data" / "low_cond.json"
UNIV = ROOT / "data" / "daily_universe.json"
PICKS = ROOT / "data" / "paper_picks.json"
MAX_AMT, POS_MIN = 37.51, 89.47

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


held = set()
if PICKS.exists():
    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    for p in picks:
        d = load(p["code"])
        if d is None:
            continue
        days = [int(x.strftime("%Y%m%d")) for x in d.index]
        if p["sig"] in days and len(days) - 1 - days.index(p["sig"]) <= 3:
            held.add(p["code"])

_u = json.loads(UNIV.read_text(encoding="utf-8"))
NAMES = {x["code"]: x["name"] for x in _u.get("items", [])}
codes = sorted(set(_u["codes"]))
rows = []
for code in codes:
    if code[-1] in PREF_TAIL:
        continue
    df = load(code)
    if df is None or len(df) < 40:
        continue
    f = features(df)
    if f is None:
        continue
    if not (MIN_AMT <= f["amt"] <= MAX_AMT) or f["pos"] < POS_MIN:
        continue
    if f["ret3"] > SURGE_MAX:
        continue
    rows.append(dict(code=code, name=NAMES.get(code, code), close=f["close"], amt=f["amt"],
                     pos=f["pos"], atr=f["atr"], ret3=f["ret3"],
                     score=cell_score(f["amt"], f["pos"], f["atr"]),
                     held=code in held))

rows.sort(key=lambda x: -x["score"])
OUTJ = ROOT / "data" / "low_preview.json"
_last = None
if rows:
    d0 = load(rows[0]["code"])
    _last = int(d0.index[-1].strftime("%Y%m%d")) if d0 is not None else None
OUTJ.write_text(json.dumps(dict(sig=_last, rows=[
    dict(name=r["name"], score=round(r["score"], 1), close=r["close"],
         amt=round(r["amt"], 1), pos=round(r["pos"], 1), atr=round(r["atr"], 1),
         ret3=round(r["ret3"], 1), held=r["held"]) for r in rows],
    buy=sum(1 for r in rows if not r["held"])), ensure_ascii=False, indent=1),
    encoding="utf-8")
print(f"[INFO] 오늘 조건 충족 {len(rows)}건 (보유중 {sum(1 for r in rows if r['held'])})")
print(f"{'점수':>6}{'종목':>14}{'종가':>10}{'거래대금':>9}{'위치':>8}{'ATR':>7}{'3일':>8}  보유")
for r in rows[:25]:
    print(f"{r['score']:>6.1f}{r['name']:>14}{r['close']:>10,.0f}{r['amt']:>8.1f}억"
          f"{r['pos']:>7.1f}%{r['atr']:>6.1f}%{r['ret3']:>+7.1f}%  {'보유' if r['held'] else ''}")
