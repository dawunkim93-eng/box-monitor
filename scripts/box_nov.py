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
DUMP = Path("/tmp/dump_nov4.json")   # --slots 999 --cool 1 (엔진이 체결일 기준 차단)
OUT = ROOT / "data" / "box_nov.json"
PIT = Path("/home/lucky/backtest/data/pit")
SINCE = 20260901
BOX_SLOT = sum([6, 6, 9, 12, 18, 18, 18]) * 10000   # 주봉박스 슬롯 87만원
LOW_SLOT = sum([6, 6, 9]) * 10000                    # 저변동 슬롯 21만원
SCALE = LOW_SLOT / BOX_SLOT                          # 비교용 환산 계수

rows = json.loads(DUMP.read_text(encoding="utf-8"))
master = pd.read_pickle(PIT / "master.pkl")
names = dict(zip(master["code"].astype(str), master["name"].astype(str)))
codes = sorted({int(r.get("code")) for r in rows if r.get("code") is not None})
idx_name = {}
# 덤프의 code는 유니버스 인덱스 → 이름 복원은 symbol 필드가 있으면 사용
sel = [r for r in rows if int(r.get("entry_day") or 0) >= SINCE]
print(f"[INFO] 9~11월 진입 {len(sel)}건 (쿨다운은 엔진에서 체결일 기준 적용)")

# 표시용: 진입일 분포와 쿨다운 차단일
_m = pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"]).set_index("Date")
_dn = [int(d.strftime("%Y%m%d")) for d in _m.index]
_ret = _m["Close"].pct_change() * 100
BLOCK = sorted({_dn[i + 1] for i in range(1, len(_dn) - 1) if _ret.iloc[i] <= -3.0
                and _dn[i + 1] >= SINCE})
_ecnt = defaultdict(int)
for r in sel:
    _ecnt[int(r["entry_day"])] += 1

def _net(r):
    return float(r.get("net") or 0) * SCALE

net_sum = sum(_net(r) for r in sel)
win = sum(1 for r in sel if _net(r) > 0)
esc = sum(1 for r in sel if r.get("escape0"))
kinds = Counter(r.get("exit") or "기타" for r in sel)

byday = defaultdict(lambda: [0, 0.0])
for r in sel:
    d = int(r.get("exit_day") or r["entry_day"])
    byday[d][0] += 1
    byday[d][1] += _net(r)
daily, cum = [], 0.0
for d in sorted(byday):
    n, net = byday[d]
    cum += net
    daily.append(dict(d=d, n=n, ein=_ecnt.get(d, 0), net=round(net), cum=round(cum),
                      pct=round(cum / LOW_SLOT * 100, 2)))

recent = sorted(sel, key=lambda r: -int(r.get("exit_day") or r["entry_day"]))[:12]
rec = []
for r in recent:
    nm = r.get("name") or names.get(str(r.get("symbol", "")), "")
    rec.append(dict(name=nm or "-", entry=int(r.get("entry_day") or 0),
                    exit=int(r.get("exit_day") or r.get("entry_day") or 0),
                    net=round(_net(r)), pct=round(_net(r) / LOW_SLOT * 100, 2),
                    kind=r.get("exit") or "",
                    cycles=r.get("cycles"), days=r.get("days")))

_s = dict(positions=len(sel), win=round(win / len(sel) * 100, 1),
          esc=round(esc / len(sel) * 100, 1),
          avg=round(net_sum / len(sel)), total=round(net_sum),
          pct=round(net_sum / LOW_SLOT * 100, 2),
          avg_pct=round(net_sum / len(sel) / LOW_SLOT * 100, 3),
          best=round(max(_net(r) for r in sel)),
          worst=round(min(_net(r) for r in sel)),
          slot=LOW_SLOT, box_slot=BOX_SLOT, scale=round(SCALE, 3),
          kinds=dict(kinds))
out = dict(
    updated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    entries_by_day=[dict(d=k, n=v) for k, v in sorted(_ecnt.items())],
    blocked=BLOCK,
    period=dict(start=min(int(r["entry_day"]) for r in sel),
                end=max(int(r.get("exit_day") or r["entry_day"]) for r in sel)),
    summary=_s,
    daily=daily, recent=rec,
    note=("슬롯 무제한 · 급락 쿨다운 적용(전일 KOSPI -3% 이하 → 당일 진입 제외) · "
          "1.5% 7차 · 당일 종가 청산 · 6주 주봉 밴드 · "
          f"금액은 슬롯 {LOW_SLOT//10000}만원 기준(주봉박스 {BOX_SLOT//10000}만원의 "
          f"{SCALE*100:.1f}%로 환산)"),
)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[INFO] 9~11월 {len(sel)}건 · 승률 {_s['win']}% · 탈출 {_s['esc']}% · "
      f"건당 {_s['avg']:,}원({_s['avg_pct']:+.3f}%) · 합계 {_s['total']:,}원({_s['pct']:+.2f}%)")
print(f"[INFO] 일별 {len(daily)}일 · 청산유형 {dict(kinds)}")
