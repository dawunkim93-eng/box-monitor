"""분봉 수집 대상 선정 → data/minute_universe.json

기준: 최근 20거래일 평균 거래대금 상위 N · 주가 1,000원 이상 · 우선주/ETF 포함
"""
import json
from pathlib import Path

import pandas as pd

PIT = Path("/home/lucky/backtest/data/pit")
OUT = Path("/home/lucky/box-monitor/data/minute_universe.json")
TOP_N = 300
MIN_AMT = 3_000_000_000      # 20일 평균 거래대금 30억
MIN_PX = 1000

master = pd.read_pickle(PIT / "master.pkl")
rows = []
for r in master.itertuples():
    code, name = str(r.code), str(r.name)
    try:
        df = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        continue
    if df is None or len(df) < 60 or "Volume" not in df.columns:
        continue
    df = df[df["Volume"] > 0]
    if len(df) < 60:
        continue
    tail = df.tail(20)
    amt = float((tail["Close"] * tail["Volume"]).mean())
    px = float(tail["Close"].iloc[-1])
    if amt < MIN_AMT or px < MIN_PX:
        continue
    rows.append((code, name, amt, px))

rows.sort(key=lambda x: -x[2])
rows = rows[:TOP_N]
out = dict(updated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
           rule=f"20일 평균 거래대금 {MIN_AMT/1e8:.0f}억 이상 · 주가 {MIN_PX:,}원 이상 · 상위 {TOP_N}",
           count=len(rows),
           codes=[c for c, *_ in rows],
           items=[dict(code=c, name=n, amt20=round(a / 1e8, 1), px=round(p))
                  for c, n, a, p in rows])
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[INFO] {len(rows)}종목 → {OUT}")
print("상위 10:", ", ".join(f"{n}({a/1e8:.0f}억)" for _, n, a, _ in rows[:10]))
print("하위 3:", ", ".join(f"{n}({a/1e8:.0f}억)" for _, n, a, _ in rows[-3:]))
