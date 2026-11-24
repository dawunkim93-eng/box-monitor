"""후보 종목 분봉 정밀 계산 — 장중 순서를 반영

일봉 모델은 고가·저가 순서를 몰라 당일 매도 판정이 부정확하다.
분봉(1분 종가)으로 체결·매도·손절 순서를 그대로 재현한다.

산출: data/minute_picks.json  {(code, sig): {status, kind, net, fill, entry, exit}}
"""
import gzip
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path("/home/lucky/box-monitor")
PICKS = ROOT / "data" / "paper_picks.json"
OUTJ = ROOT / "data" / "minute_picks.json"
MDIR = ROOT / "data" / "minute"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

BUY_GAP, SELL_GAP, FEE, AUTO_STOP = 0.015, 0.020, 0.00015, 0.035
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3


def fetch(code, count=3000):
    url = (f"https://fchart.stock.naver.com/sise.nhn?symbol={code}"
           f"&timeframe=minute&count={count}&requestType=0")
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) \
        .read().decode("euc-kr", "ignore")
    rows = []
    for it in re.findall(r'<item data="([^"]+)"', raw):
        p = it.split("|")
        if len(p) < 6:
            continue
        ts = p[0]
        try:
            c = float(p[4])
        except ValueError:
            continue
        if c > 0 and len(ts) == 12:
            rows.append((ts, c))
    rows.sort()
    return rows


def cached(code):
    p = MDIR / f"{code}.csv.gz"
    if not p.exists():
        return []
    with gzip.open(p, "rt") as f:
        rows = [(t[0], float(t[1])) for t in
                (line.strip().split(",") for line in f)
                if len(t) == 3 and t[0].isdigit()]
    rows.sort()
    return rows


def evaluate(rows, entry_day, prev_close):
    """분봉으로 매수·매도·손절 순서 재현"""
    bars = [r for r in rows if int(r[0][:8]) >= entry_day]
    if not bars:
        return None
    day = int(bars[0][0][:8])
    today = [(t, p) for t, p in bars if int(t[:8]) == day]
    if not today:
        return None
    px = None
    lv = tg = stop_px = limit_entry = None
    held, net, kind, exit_day = {}, 0.0, None, None
    days = sorted({int(t[:8]) for t, _ in bars})[:HOLD]
    for di, d in enumerate(days):
        seq = [(t, p) for t, p in bars if int(t[:8]) == d]
        for ts, p in seq:
            hhmm = ts[8:]
            # 1) 매수
            if px is None:
                if p < prev_close:
                    px = p
                    limit_entry = False
                elif p <= prev_close:
                    px = prev_close
                    limit_entry = True
                else:
                    continue
                lv = [px * (1 - BUY_GAP) ** k for k in range(3)]
                tg = [x * (1 + SELL_GAP) for x in lv]
                stop_px = px * (1 - AUTO_STOP)
                held = {0: px}
                continue
            # 2) 차수 체결
            for k in (1, 2):
                if k not in held and p <= lv[k]:
                    held[k] = lv[k]
            # 3) 매도
            for k in list(held):
                if p >= tg[k]:
                    net += AMTS[k] * 10_000 * (SELL_GAP - FEE)
                    del held[k]
                    exit_day = d
            if not held:
                kind = kind or "전량익절"
                return dict(status="청산완료", kind=kind, net=net / SLOT * 100,
                            fill=round(px), entry=d, exit=exit_day)
            # 4) 자동손절
            if p <= stop_px:
                for k in list(held):
                    net += AMTS[k] * 10_000 * (p / lv[k] - 1 - FEE)
                return dict(status="청산완료", kind="자동 손절", net=net / SLOT * 100,
                            fill=round(px), entry=d, exit=d)
        # 5) 종가(장 마감) 3차 이탈
        last = seq[-1][1]
        if held and last < lv[0] * (1 - BUY_GAP) ** 2:
            for k in list(held):
                net += AMTS[k] * 10_000 * (last / lv[k] - 1 - FEE)
            return dict(status="청산완료", kind="손절선 청산", net=net / SLOT * 100,
                        fill=round(px), entry=days[0], exit=d)
    if px is None:
        return dict(status="미체결", kind="미체결", net=0.0, fill=None, entry=None, exit=None)
    last_d = days[-1]
    last_px = [p for t, p in bars if int(t[:8]) == last_d][-1]
    for k in list(held):
        net += AMTS[k] * 10_000 * (last_px / lv[k] - 1 - FEE)
    done = len(days) >= HOLD
    return dict(status="청산완료" if done else "보유중",
                kind=("만기 청산" if done else "보유중"),
                net=net / SLOT * 100, fill=round(px), entry=days[0], exit=last_d)


def main():
    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    out = json.loads(OUTJ.read_text(encoding="utf-8")) if OUTJ.exists() else {}
    n = 0
    for p in picks:
        key = f"{p['code']}:{p['sig']}"
        if key in out:
            continue
        rows = cached(p["code"])
        _mx = max((int(t[:8]) for t, _ in rows), default=0)
        if not rows or _mx < p["sig"]:          # 캐시가 신호일 이후를 못 덮으면 새로 받음
            fresh = fetch(p["code"])
            if fresh:
                rows = fresh
        if not rows:
            continue
        # 신호일 다음 거래일 = 분봉상 신호일 이후 첫 날
        sig = p["sig"]
        days = sorted({int(t[:8]) for t, _ in rows})
        later = [d for d in days if d > sig]
        if not later:
            continue
        prev_close = float(p["limit"])
        r = evaluate(rows, later[0], prev_close)
        if r is None:
            continue
        out[key] = r
        n += 1
    OUTJ.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[INFO] 분봉 계산 {n}건 → {OUTJ.name} (누적 {len(out)})")


if __name__ == "__main__":
    main()
