"""PIT 최근 봉 보정 — 네이버 일봉으로 마지막 N거래일을 덮어쓴다

이유: pit_download.py는 KRX 로그인이 필요해 실패하면 종가가 잠정값으로 남는다.
      (장 마감 직후 수집분은 종가·거래량이 확정 전 값)
사용: python scripts/pit_naver_fix.py [--days 15] [--codes ...]
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

PIT = Path("/home/lucky/backtest/data/pit")
UNIV = Path("/home/lucky/box-monitor/data/daily_universe.json")
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(code, days):
    end = pd.Timestamp.now()
    st = (end - pd.Timedelta(days=days * 2 + 10)).strftime("%Y%m%d")
    url = (f"https://api.finance.naver.com/siseJson.naver?symbol={code}"
           f"&requestType=1&startTime={st}&endTime={end:%Y%m%d}&timeframe=day")
    req = urllib.request.Request(url, headers=UA)
    raw = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", "ignore")
    rows = re.findall(r'\["(\d{8})",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*(\d+)', raw)
    out = {}
    for d, o, h, l, c, v in rows:
        out[pd.Timestamp(d)] = (float(o), float(h), float(l), float(c), float(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=15)
    ap.add_argument("--codes", nargs="*")
    a = ap.parse_args()

    codes = a.codes or json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
    fixed = ok = 0
    for i, code in enumerate(codes, 1):
        p = PIT / f"{code}.pkl"
        if not p.exists():
            continue
        try:
            new = fetch(code, a.days)
        except Exception:
            continue
        if not new:
            continue
        try:
            df = pd.read_pickle(p)
        except Exception:
            continue
        before = df.copy()
        cols = {k: [] for k in ("Open", "High", "Low", "Close", "Volume")}
        idx = []
        for d, (o, h, l, c, v) in sorted(new.items()):
            idx.append(d)
            cols["Open"].append(o); cols["High"].append(h)
            cols["Low"].append(l); cols["Close"].append(c); cols["Volume"].append(v)
        add = pd.DataFrame(cols, index=pd.DatetimeIndex(idx))
        df = df[~df.index.isin(add.index)]
        df = pd.concat([df, add]).sort_index()
        if len(df) != len(before) or not df.tail(1).equals(before.tail(1)):
            df.to_pickle(p)
            fixed += 1
        ok += 1
        if i % 100 == 0:
            print(f"  {i}/{len(codes)} · 갱신 {fixed}", flush=True)
    print(f"[INFO] 확인 {ok}종목 · 갱신 {fixed}종목 (최근 {a.days}거래일)")


if __name__ == "__main__":
    main()
