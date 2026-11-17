"""주봉 박스 검증 누적 곡선 → data/box_curve.json

입력:
  /tmp/dump_cur.json          — box_validation.py --dump (포지션별 entry/exit/net)
  data/box_validation.json    — 연도별 요약
  magicsplit results/box_ledger.jsonl — 실거래(신호) 원장
산출:
  daily   [{d, n, net, cum}]   일별 실현 손익 누적
  monthly [{m, n, net, cum}]   월별
  yearly  [{year, n, net, cum, esc}]
  live    실거래 진입-청산 매칭 누적
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
DUMP = Path("/tmp/dump_cur.json")
OUT = ROOT / "data" / "box_curve.json"
VAL = ROOT / "data" / "box_validation.json"
LEDGER = Path("/home/lucky/magicsplit-safe-strategy/backtest/results/box_ledger.jsonl")

rows = json.loads(DUMP.read_text(encoding="utf-8"))
print(f"[INFO] 포지션 {len(rows):,}")

byday = defaultdict(lambda: [0, 0.0])
for r in rows:
    d = int(r.get("exit_day") or r["entry_day"])
    byday[d][0] += 1
    byday[d][1] += float(r.get("net") or 0)

daily, cum, peak, mdd = [], 0.0, 0.0, 0.0
for d in sorted(byday):
    n, net = byday[d]
    cum += net
    peak = max(peak, cum)
    mdd = min(mdd, cum - peak)
    daily.append(dict(d=d, n=n, net=round(net), cum=round(cum)))

# 월별
bym = defaultdict(lambda: [0, 0.0])
for x in daily:
    m = x["d"] // 100
    bym[m][0] += x["n"]
    bym[m][1] += x["net"]
monthly, mc = [], 0.0
for m in sorted(bym):
    n, net = bym[m]
    mc += net
    monthly.append(dict(m=m, n=n, net=round(net), cum=round(mc)))

vy = {y["year"]: y for y in json.loads(VAL.read_text(encoding="utf-8"))["yearly"]}
yearly, yc = [], 0.0
for y in sorted(vy):
    yc += vy[y]["net"]
    yearly.append(dict(year=y, n=vy[y]["positions"], net=round(vy[y]["net"]),
                       cum=round(yc), esc=vy[y]["esc_rate"]))

live = []   # 원장은 청산가가 비어 있는 건이 많아 손익 복원 제외

out = dict(updated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
           source=dict(positions=len(rows), days=len(daily),
                       first=daily[0]["d"] if daily else None,
                       last=daily[-1]["d"] if daily else None,
                       mdd=round(mdd), peak=round(peak)),
           daily=daily, monthly=monthly, yearly=yearly, live=live)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
print(f"[INFO] 일별 {len(daily)}일 · 월별 {len(monthly)}개월 · 연도 {len(yearly)} · 실거래 {len(live)}건")
print(f"[INFO] 최종 누적 {daily[-1]['cum']:,}원 · MDD {mdd:,.0f}원 · 피크 {peak:,.0f}원")

