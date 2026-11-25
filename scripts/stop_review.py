"""최근 2주 자동손절 종목 복기 — 분봉으로 회복 여부 확인

손절(1차 매수가 -3.5%)이 실제로 맞는 판단이었는지 본다.
각 종목의 손절 시점 이후 주가가 다시 1차 매수가+2%(익절선)까지 올라왔는지 확인.
"""
import gzip
from pathlib import Path

ROOT = Path("/home/lucky/box-monitor")

# 사용자 실제 매매에서 자동손절로 끝난 종목 (11/23~11/25)
CASES = [
    # (종목명, 코드, 매수일, 1차 매수가, 손절일)
    ("제닉스로보틱스", "381620", "20261124", 6280, "20261125"),
    ("다원넥스뷰", "323350", "20261125", 14850, "20261125"),
    ("케이씨", "029460", "20261125", 46000, "20261125"),
    ("LIG아큐버", "073490", "20261125", 34200, "20261125"),
    ("청담글로벌", "362320", "20261124", 5580, "20261124"),
    ("KTcs", "058850", "20261120", 2305, "20261123"),
    ("대웅", "003090", "20261123", 16250, "20261123"),
    ("케이뱅크", "279570", "20261118", 5430, "20261123"),
]
STOP = 0.035
TGT = 0.020


def load(code):
    p = ROOT / "data" / "minute" / f"{code}.csv.gz"
    if not p.exists():
        return []
    with gzip.open(p, "rt") as f:
        out = []
        for line in f:
            t = line.strip().split(",")
            if len(t) == 3 and t[0].isdigit():
                out.append((t[0], float(t[1])))
    return out


rows = []
for name, code, buy_day, entry, stop_day in CASES:
    bars = load(code)
    if not bars:
        rows.append(dict(name=name, note="분봉 없음"))
        continue
    stop_px = entry * (1 - STOP)
    tgt_px = entry * (1 + TGT)
    # 손절일 당일 봉
    day = [b for b in bars if b[0].startswith(stop_day)]
    if not day:
        rows.append(dict(name=name, note=f"{stop_day} 분봉 없음"))
        continue
    # 손절 시점 = 처음으로 종가가 손절선 아래로 내려간 분
    hit_i = next((i for i, (t, p) in enumerate(day) if p <= stop_px), None)
    if hit_i is None:
        rows.append(dict(name=name, note="손절선 미터치(데이터 확인 필요)"))
        continue
    after = day[hit_i:]
    post_high = max(p for _, p in after)
    post_low = min(p for _, p in after)
    day_close = day[-1][1]
    # 이후 거래일 (다음 3거래일)
    later = [b for b in bars if b[0][:8] > stop_day]
    later_days = sorted({b[0][:8] for b in later})[:3]
    later_high = max((p for t, p in later if t[:8] in later_days), default=None)
    later_close = [p for t, p in later if t[:8] == later_days[-1]] if later_days else []
    # -4% 손절선 시나리오: 4%선을 안 건드리고 +2%에 닿았는가
    stop4 = entry * (1 - 0.04)
    hit4 = next((i for i, (t, p) in enumerate(day) if p <= stop4), None)
    if hit4 is None:
        after4 = day
        sc4 = "미터치 → 익절 도달" if max(p for _, p in after4) >= tgt_px else "미터치"
    else:
        sc4 = "4%도 손절"
    rows.append(dict(
        name=name, entry=entry, stop_px=round(stop_px), tgt_px=round(tgt_px),
        hit_time=day[hit_i][0][8:12], hit_px=day[hit_i][1],
        post_high=post_high, post_low=post_low, day_close=day_close,
        recovered_tgt=post_high >= tgt_px, sc4=sc4,
        later_days=",".join(later_days), later_high=later_high,
        later_recover=(later_high is not None and later_high >= tgt_px),
        later_close=(later_close[-1] if later_close else None)))

print(f"{'종목':<12}{'1차':>8}{'손절선':>8}{'터치시각':>8}{'터치가':>8}"
      f"{'이후최저':>9}{'이후최고':>9}{'익절선':>8}{'+2%도달':>8}{'4% 시나리오':>16}")
for r in rows:
    if "note" in r:
        print(f"{r['name']:<12} {r['note']}")
        continue
    print(f"{r['name']:<12}{r['entry']:>8,}{r['stop_px']:>8,}{r['hit_time']:>8}"
          f"{r['hit_px']:>8,.0f}{r['post_low']:>9,.0f}{r['post_high']:>9,.0f}{r['tgt_px']:>8,}"
          f"{'O' if r['recovered_tgt'] else 'X':>8}{r['sc4']:>16}")

print()
for r in rows:
    if "note" in r:
        continue
    lh = r["later_high"]
    tail = f"· 이후 최고 {lh:,.0f}원 ({lh/r['entry']-1:+.2%})" if lh else "· 이후 데이터 없음"
    print(f"{r['name']}: 손절 {r['hit_time'][:2]}:{r['hit_time'][2:]} {r['hit_px']:,.0f}원 "
          f"→ 당일 종가 {r['day_close']:,.0f}원 ({r['day_close']/r['entry']-1:+.2%}) {tail}")
