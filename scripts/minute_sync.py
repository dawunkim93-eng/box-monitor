"""네이버 분봉 수집 → data/minute/{code}.csv.gz 누적.

사용:
  python3 scripts/minute_sync.py            # 관찰 종목(박스 라이브) 전체, 최근 8일 백필
  python3 scripts/minute_sync.py --codes 005930,000660
  python3 scripts/minute_sync.py --no-backfill   # 오늘 것만

저장 형식: timestamp(YYYYMMDDHHMM), close, volume  (중복 시각은 버림)
네이버 fchart는 1분봉 종가·거래량만 제공합니다(OHLC 없음).
"""
import argparse
import gzip
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "minute"
LIVE = ROOT / "data" / "box_live.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch(code, count=3000):
    url = (f"https://fchart.stock.naver.com/sise.nhn?symbol={code}"
           f"&timeframe=minute&count={count}&requestType=0")
    req = urllib.request.Request(url, headers=UA)
    raw = urllib.request.urlopen(req, timeout=20).read().decode("euc-kr", "ignore")
    rows = []
    for it in re.findall(r'<item data="([^"]+)"', raw):
        p = it.split("|")
        if len(p) < 6:
            continue
        ts = p[0]
        try:
            c = float(p[4])
            v = float(p[5])
        except ValueError:
            continue
        if c > 0 and len(ts) == 12:
            rows.append((ts, c, v))
    rows.sort()
    return rows


def load(code):
    p = OUT / f"{code}.csv.gz"
    if not p.exists():
        return {}
    with gzip.open(p, "rt") as f:
        out = {}
        for line in f:
            t = line.strip().split(",")
            if len(t) == 3 and t[0].isdigit():
                out[t[0]] = (t[1], t[2])
        return out


def save(code, data):
    p = OUT / f"{code}.csv.gz"
    with gzip.open(p, "wt") as f:
        for ts in sorted(data):
            f.write(f"{ts},{data[ts][0]},{data[ts][1]}\n")


def watchlist():
    if not LIVE.exists():
        return []
    d = json.loads(LIVE.read_text(encoding="utf-8"))
    codes = []
    for r in d.get("rows", []):
        c = str(r.get("code") or "")
        if len(c) == 6 and c not in codes:
            codes.append(c)
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="", help="쉼표로 구분한 종목코드 (기본: 박스 라이브 전체)")
    ap.add_argument("--count", type=int, default=3000, help="가져올 분봉 개수 (최대 3000 ≈ 8거래일)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    codes = ([c.strip() for c in args.codes.split(",") if c.strip()]
             if args.codes else watchlist())
    if not codes:
        print("[WARN] 대상 종목 없음 (data/box_live.json 확인)")
        return 1

    tot_new = 0
    ok = fail = 0
    for i, code in enumerate(codes, 1):
        try:
            fresh = fetch(code, args.count)
        except Exception as e:
            fail += 1
            print(f"  [{i}/{len(codes)}] {code} 실패 {e}")
            time.sleep(0.3)
            continue
        cur = load(code)
        before = len(cur)
        for ts, c, v in fresh:
            cur[ts] = (c, v)
        new = len(cur) - before
        save(code, cur)
        tot_new += new
        ok += 1
        if i % 10 == 0 or i == len(codes):
            print(f"  [{i}/{len(codes)}] {code} +{new}봉 · 누적 {len(cur):,}", flush=True)
        time.sleep(0.25)

    files = list(OUT.glob("*.csv.gz"))
    days = set()
    for f in files:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                days.add(line[:8])
    print(f"[INFO] 성공 {ok} · 실패 {fail} · 신규 {tot_new:,}봉 · "
          f"파일 {len(files)}종목 · 수집일 {len(days)}일 ({min(days) if days else '-'}~{max(days) if days else '-'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
