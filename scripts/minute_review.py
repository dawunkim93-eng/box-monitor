"""분봉 실측 리포트 → data/minute_review.json

목표 모델 (사용자 지정):
- 종목 검색: 전일 종가 기준
- 진입: 전일 종가 지정가 (시가가 더 낮으면 그 가격에 체결)
- 목표: 당일 탈출(+1.5%) · 안 되면 최대 5거래일 추적 후 종가 평가

측정:
- 탈출율 = 1차가 +1.5%에 팔린 비율
- 탈출일 = 진입일 기준 몇 번째 거래일에 팔렸는지 (당일 = 1)
- 사이클 = 보유 기간 중 왕복 횟수
- 건당 수익 = 슬롯 87만원 대비 %
"""
import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN = ROOT / "data" / "minute"
PIT = Path("/home/lucky/backtest/data/pit")
OUT = ROOT / "data" / "minute_review.json"

GAP = 0.015
AMTS = [6, 6, 9, 12, 18, 18, 18]
SLOT = sum(AMTS) * 10_000
FEE = 0.00015
MAX_DAYS = 5


def load_bars(p):
    by_day = defaultdict(list)
    with gzip.open(p, "rt") as f:
        for line in f:
            t = line.strip().split(",")
            if len(t) != 3 or not t[0].isdigit():
                continue
            try:
                by_day[t[0][:8]].append((t[0][8:], float(t[1])))
            except ValueError:
                continue
    for d in by_day:
        by_day[d].sort()
    return by_day


def sim_position(days, bars, limit):
    levels = [limit * (1 - GAP) ** k for k in range(7)]
    held = [True] + [False] * 6
    net = 0.0
    cycles = 0
    escape_day = None
    worst = 0.0
    last_px = limit
    for di, day in enumerate(days):
        seq = bars.get(day) or []
        if not seq:
            continue
        last_px = seq[-1][1]
        for _, c in seq:
            for k in range(1, 7):
                if not held[k] and c <= levels[k]:
                    held[k] = True
            for k in range(7):
                if not held[k]:
                    continue
                if c >= levels[k] * (1 + GAP):
                    held[k] = False
                    cycles += 1
                    net += AMTS[k] * 10_000 * (GAP - FEE)
                    if k == 0 and escape_day is None:
                        escape_day = di
            val = sum(AMTS[k] * 10_000 * (c / levels[k] - 1)
                      for k in range(7) if held[k])
            worst = min(worst, val / SLOT * 100)
    for k in range(7):
        if held[k]:
            net += AMTS[k] * 10_000 * (last_px / levels[k] - 1)
    return dict(net=net, cycles=cycles, escape_day=escape_day, worst=worst)


def main():
    files = sorted(MIN.glob("*.csv.gz"))
    if not files:
        print("[WARN] data/minute 비어 있음 — minute_sync 먼저 실행")
        return

    per = defaultdict(lambda: dict(pos=0, esc=0, net=0.0, cyc=0, worst=0.0,
                                   days_sum=0, same=0))
    dist = defaultdict(int)
    recent = []
    all_days = set()

    for p in files:
        code = p.name.replace(".csv.gz", "")
        bars = load_bars(p)
        days = sorted(bars)
        all_days.update(days)
        for i in range(len(days) - 1):
            limit = bars[days[i]][-1][1]                  # 신호일 종가 = 지정가
            seq_days = days[i + 1:i + 1 + MAX_DAYS]
            first = bars.get(seq_days[0]) or []
            if not first or min(c for _, c in first) > limit:
                continue                                  # 미체결
            r = sim_position(seq_days, bars, limit)
            s = per[code]
            s["pos"] += 1
            s["net"] += r["net"]
            s["cyc"] += r["cycles"]
            s["worst"] = min(s["worst"], r["worst"])
            if r["escape_day"] is not None:
                s["esc"] += 1
                s["days_sum"] += r["escape_day"] + 1
                if r["escape_day"] == 0:
                    s["same"] += 1
                dist["당일" if r["escape_day"] == 0 else f"{r['escape_day'] + 1}일"] += 1
            else:
                dist[f"{MAX_DAYS}일 내 미탈출"] += 1
            recent.append(dict(code=code, sig=days[i], ret=round(r["net"] / SLOT * 100, 3),
                               esc=(None if r["escape_day"] is None else r["escape_day"] + 1),
                               cyc=r["cycles"], worst=round(r["worst"], 2)))

    names = {}
    try:
        import pandas as pd
        master = pd.read_pickle(PIT / "master.pkl")
        names = dict(zip(master["code"].astype(str), master["name"].astype(str)))
    except Exception:
        pass

    stocks = []
    for code, s in per.items():
        n = s["pos"]
        if n < 2:
            continue
        stocks.append(dict(
            code=code, name=names.get(code, code), pos=n,
            esc=round(s["esc"] / n * 100, 1),
            same=round(s["same"] / n * 100, 1),
            days=round(s["days_sum"] / max(s["esc"], 1), 2),
            cyc=round(s["cyc"] / n, 1),
            ret=round(s["net"] / SLOT * 100 / n, 3),
            worst=round(s["worst"], 2)))
    stocks.sort(key=lambda x: (-x["ret"],))

    tot = sum(x["pos"] for x in stocks)
    esc_n = sum(x["esc"] * x["pos"] / 100 for x in stocks)
    same_n = sum(x["same"] * x["pos"] / 100 for x in stocks)
    days_avg = (sum(x["days"] * (x["esc"] * x["pos"] / 100) for x in stocks)
                / max(esc_n, 1))

    out = dict(
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        window=dict(start=min(all_days) if all_days else None,
                    end=max(all_days) if all_days else None,
                    days=len(all_days), stocks=len(stocks)),
        summary=dict(positions=int(tot),
                     esc_rate=round(esc_n / max(tot, 1) * 100, 1),
                     same_day=round(same_n / max(tot, 1) * 100, 1),
                     days_to_esc=round(days_avg, 2),
                     ret=round(sum(x["ret"] * x["pos"] for x in stocks) / max(tot, 1), 3),
                     dist=dict(dist)),
        stocks=stocks,
        recent=sorted(recent, key=lambda x: (x["sig"], -x["ret"]))[-400:],
        assumptions=dict(gap=GAP * 100, amts=AMTS, slot=SLOT, max_days=MAX_DAYS,
                         entry="전일 종가 지정가 (시가가 더 낮으면 시가 체결)",
                         exit="차수×1.015 분봉 종가 · 미탈출 시 5일째 종가 평가",
                         note="분봉 종가 기준 · 고가/저가 미제공"),
    )
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    s = out["summary"]
    print(f"[INFO] {out['window']} · 포지션 {s['positions']:,} · 탈출 {s['esc_rate']}%"
          f"(당일 {s['same_day']}%) · 평균 {s['days_to_esc']}일 · 건당 {s['ret']}% · {s['dist']}")


if __name__ == "__main__":
    main()
