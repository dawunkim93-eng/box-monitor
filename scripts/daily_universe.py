"""저유동(사다리) 전략용 유니버스 → data/daily_universe.json

기준: 20일 평균 거래대금 3억~60억 · 주가 1,000원 이상 · 60일 이상 데이터
(2년 검증의 최적 구간 37.5억 이하를 넉넉히 감싸도록 범위를 잡음)
"""
import json
from pathlib import Path

import pandas as pd

PIT = Path("/home/lucky/backtest/data/pit")
OUT = Path("/home/lucky/box-monitor/data/daily_universe.json")
AMT_LO, AMT_HI = 3.0, 60.0        # 억원
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
    amt = float((tail["Close"] * tail["Volume"]).mean()) / 1e8
    px = float(tail["Close"].iloc[-1])
    if not (AMT_LO <= amt <= AMT_HI) or px < MIN_PX:
        continue
    rows.append((code, name, amt, px))

rows.sort(key=lambda x: -x[2])
out = dict(updated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
           rule=f"20일 평균 거래대금 {AMT_LO:.0f}~{AMT_HI:.0f}억 · 주가 {MIN_PX:,}원 이상",
           count=len(rows),
           codes=[c for c, *_ in rows],
           items=[dict(code=c, name=n, amt20=round(a, 1), px=round(p))
                  for c, n, a, p in rows])
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[INFO] {len(rows)}종목 → {OUT}")
print("상위:", ", ".join(f"{n}({a:.0f}억)" for _, n, a, _ in rows[:5]))
print("하위:", ", ".join(f"{n}({a:.0f}억)" for _, n, a, _ in rows[-5:]))
