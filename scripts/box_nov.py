"""주봉박스 11월 실적 (슬롯 무제한) → data/box_nov.json

저변동 모의매매와 같은 양식으로 복기 탭에 표시하기 위한 데이터.
  - 기간: 2026-11-01 이후 진입
  - 슬롯 무제한(--slots 999) 가정
  - 청산 유형·일별 누적·최근 기록
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
DUMP = Path("/tmp/dump_nov.json")
OUT = ROOT / "data" / "box_nov.json"
PIT = Path("/home/lucky/backtest/data/pit")
SINCE = 20261101

rows = json.loads(DUMP.read_text(encoding="utf-8"))
master = pd.read_pickle(PIT / "master.pkl")
names = dict(zip(master["code"].astype(str), master["name"].astype(str)))
codes = sorted({int(r.get("code")) for r in rows if r.get("code") is not None})
idx_name = {}
# 덤프의 code는 유니버스 인덱스 → 이름 복원은 symbol 필드가 있으면 사용
sel = [r for r in rows if int(r.get("entry_day") or 0) >= SINCE]
print(f"[INFO] 전체 {len(rows):,}건 중 11월 진입 {len(sel):,}건")

if not sel:
    OUT.write_text(json.dumps(dict(updated_at=str(pd.Timestamp.now())[:16], empty=True),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("[WARN] 11월 데이터 없음")
    raise SystemExit

net_sum = sum(float(r.get("net") or 0) for r in sel)
win = sum(1 for r in sel if float(r.get("net") or 0) > 0)
esc = sum(1 for r in sel if r.get("escape0"))
kinds = Counter(r.get("exit") or "기타" for r in sel)

byday = defaultdict(lambda: [0, 0.0])
for r in sel:
    d = int(r.get("exit_day") or r["entry_day"])
    byday[d][0] += 1
    byday[d][1] += float(r.get("net") or 0)
daily, cum = [], 0.0
for d in sorted(byday):
    n, net = byday[d]
    cum += net
    daily.append(dict(d=d, n=n, net=round(net), cum=round(cum)))

recent = sorted(sel, key=lambda r: -int(r.get("exit_day") or r["entry_day"]))[:12]
rec = []
for r in recent:
    nm = r.get("name") or names.get(str(r.get("symbol", "")), "")
    rec.append(dict(name=nm or "-", entry=int(r.get("entry_day") or 0),
                    exit=int(r.get("exit_day") or r.get("entry_day") or 0),
                    net=round(float(r.get("net") or 0)), kind=r.get("exit") or "",
                    cycles=r.get("cycles"), days=r.get("days")))

_s = dict(positions=len(sel), win=round(win / len(sel) * 100, 1),
          esc=round(esc / len(sel) * 100, 1),
          avg=round(net_sum / len(sel)), total=round(net_sum),
          best=round(max(float(r.get("net") or 0) for r in sel)),
          worst=round(min(float(r.get("net") or 0) for r in sel)),
          kinds=dict(kinds))
out = dict(
    updated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    period=dict(start=min(int(r["entry_day"]) for r in sel),
                end=max(int(r.get("exit_day") or r["entry_day"]) for r in sel)),
    summary=_s,
    daily=daily, recent=rec,
    note="슬롯 무제한 가정 · 1.5% 7차 · 당일 종가 청산 · 6주 주봉 밴드",
)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[INFO] 11월 {len(sel)}건 · 승률 {_s['win']}% · 탈출 {_s['esc']}% · "
      f"건당 {_s['avg']:,}원 · 합계 {_s['total']:,}원")
print(f"[INFO] 일별 {len(daily)}일 · 청산유형 {dict(kinds)}")
