"""분봉으로 사다리 실측 리포트 → data/minute_review.json.

가정:
- 진입: 그날 시가(일봉 Open). 없으면 첫 분봉 종가.
- 사다리: 1~7차, 1.5% 간격, 금액 6·6·9·12·18·18·18만원 (슬롯 87만원)
- 매도: 차수 가격 × 1.015 에 1분봉 종가가 닿으면 체결 (실제 순서대로)
- 청산 규칙은 반영하지 않음 — "사다리만 돌렸을 때"의 상한/하한을 보는 지표
"""
import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MIN = ROOT / "data" / "minute"
PIT = Path("/home/lucky/backtest/data/pit")
OUT = ROOT / "data" / "minute_review.json"

GAP = 0.015
AMTS = [6, 6, 9, 12, 18, 18, 18]      # 만원
SLOT = sum(AMTS) * 10_000            # 87만원
FEE = 0.00015


def minute_paths():
    return sorted(MIN.glob("*.csv.gz"))


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


def daily_open(code, day):
    try:
        df = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        return None
    dt = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}")
    if dt not in df.index:
        return None
    v = float(df.loc[dt, "Open"])
    return v if v > 0 else None


def sim_day(bars, entry):
    levels = [entry * (1 - GAP) ** k for k in range(7)]
    held = [True] + [False] * 6
    net = 0.0
    cycles = 0
    fills = 0
    escape = False
    worst = 0.0          # 미실현 손실 최대 (%)
    for _, c in bars:
        for k in range(1, 7):
            if not held[k] and c <= levels[k]:
                held[k] = True
                fills += 1
        for k in range(7):
            if not held[k]:
                continue
            if c >= levels[k] * (1 + GAP):
                held[k] = False
                cycles += 1
                net += AMTS[k] * 10_000 * (GAP - FEE)
                if k == 0:
                    escape = True
        # 평가손익
        val = sum(AMTS[k] * 10_000 * (c / levels[k] - 1) for k in range(7) if held[k])
        worst = min(worst, val / SLOT * 100)
    return dict(net=net, cycles=cycles, fills=fills, escape=escape, worst=worst)


def main():
    files = minute_paths()
    if not files:
        print("[WARN] data/minute 비어 있음 — minute_sync 먼저 실행")
        return

    per_stock = defaultdict(lambda: dict(days=0, net=0.0, win=0, cyc=0, esc=0,
                                         worst=0.0, fills=0))
    daily = []
    days_all = set()
    for p in files:
        code = p.name.replace(".csv.gz", "")
        by_day = load_bars(p)
        for day, bars in sorted(by_day.items()):
            if len(bars) < 30:
                continue
            entry = daily_open(code, day) or bars[0][1]
            r = sim_day(bars, entry)
            days_all.add(day)
            s = per_stock[code]
            s["days"] += 1
            s["net"] += r["net"]
            s["win"] += 1 if r["net"] > 0 else 0
            s["cyc"] += r["cycles"]
            s["esc"] += 1 if r["escape"] else 0
            s["fills"] += r["fills"]
            s["worst"] = min(s["worst"], r["worst"])
            daily.append(dict(code=code, day=day, ret=round(r["net"] / SLOT * 100, 3),
                              cycles=r["cycles"], fills=r["fills"],
                              escape=r["escape"], worst=round(r["worst"], 2)))

    names = {}
    try:
        master = pd.read_pickle(PIT / "master.pkl")
        names = dict(zip(master["code"].astype(str), master["name"].astype(str)))
    except Exception:
        pass

    stocks = []
    for code, s in per_stock.items():
        n = s["days"]
        stocks.append(dict(code=code, name=names.get(code, code), days=n,
                           tot_ret=round(s["net"] / SLOT * 100, 2),
                           day_ret=round(s["net"] / SLOT * 100 / n, 3),
                           win=round(s["win"] / n * 100, 1),
                           cyc=round(s["cyc"] / n, 1),
                           esc=round(s["esc"] / n * 100, 1),
                           fills=round(s["fills"] / n, 1),
                           worst=round(s["worst"], 2)))
    stocks.sort(key=lambda x: -x["day_ret"])

    tot_ret = sum(x["tot_ret"] for x in stocks)
    out = dict(
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        window=dict(start=min(days_all) if days_all else None,
                    end=max(days_all) if days_all else None,
                    days=len(days_all), stocks=len(stocks)),
        summary=dict(avg_day_ret=round(tot_ret / max(len(days_all), 1) / max(len(stocks), 1), 3),
                     tot_ret=round(tot_ret, 2),
                     win_rate=round(sum(x["win"] * x["days"] for x in stocks)
                                    / max(sum(x["days"] for x in stocks), 1), 1),
                     cycles=round(sum(x["cyc"] * x["days"] for x in stocks)
                                  / max(sum(x["days"] for x in stocks), 1), 1),
                     escape=round(sum(x["esc"] * x["days"] for x in stocks)
                                  / max(sum(x["days"] for x in stocks), 1), 1)),
        stocks=stocks,
        daily=sorted(daily, key=lambda x: (x["day"], -x["ret"]))[-400:],
        assumptions=dict(gap=GAP * 100, amts=AMTS, slot=SLOT,
                         entry="당일 시가", exit="차수×1.015 (분봉 종가 기준)",
                         note="청산 규칙 미적용 · 사다리만 돌린 상한"),
    )
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[INFO] {out['window']} · 종목 {len(stocks)} · 평균 일수익률 {out['summary']['avg_day_ret']}% "
          f"· 승률 {out['summary']['win_rate']}% · 사이클 {out['summary']['cycles']}")


if __name__ == "__main__":
    main()
