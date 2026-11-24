"""재료 성격(실적/계약/테마)별 성과 비교

재료일(하루 +10% & 거래량 5배)에 어떤 공시가 붙었는지로 나눠
조정 매수 전략의 성과를 비교한다.
"""
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
FEE = 0.00015

EARN = ("분기보고서", "반기보고서", "사업보고서", "영업(잠정)실적", "잠정실적",
        "매출액또는손익구조", "실적공시", "연결재무제표")
DEAL = ("단일판매", "공급계약", "수주", "판매ㆍ공급계약", "공급계약체결")
CAP = ("유상증자", "무상증자", "전환사채", "신주인수권", "합병", "분할", "최대주주")


def kind_of(names):
    txt = " ".join(names)
    if any(k in txt for k in EARN):
        return "실적"
    if any(k in txt for k in DEAL):
        return "계약"
    if any(k in txt for k in CAP):
        return "자본변동"
    return "기타/테마"


def sim(o, h, l, c, e, tgt, stop, maxd):
    px = float(o[e])
    if px <= 0:
        return None
    for d in range(maxd):
        j = e + d
        if j >= len(c):
            return None
        if h[j] >= px * (1 + tgt):
            return tgt - FEE
        if l[j] <= px * (1 - stop):
            return -stop - FEE
    j = min(e + maxd - 1, len(c) - 1)
    return float(c[j]) / px - 1 - FEE


sigs = json.loads((ROOT / "data" / "surge_signals.json").read_text(encoding="utf-8"))
disc = json.loads((ROOT / "data" / "dart_disclosures.json").read_text(encoding="utf-8"))

# 종목별 공시 날짜 인덱스
by_code = {}
for code, rows in disc.items():
    by_code[code] = [(r["date"], r["name"]) for r in rows]

rows = []
cache = {}
for s in sigs:
    code, day = s["code"], s["day"]
    if code not in cache:
        cache[code] = by_code.get(code, [])
    lst = cache[code]
    d0 = datetime.strptime(str(day), "%Y%m%d")
    lo = (d0 - timedelta(days=7)).strftime("%Y%m%d")
    hi = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    names = [n for dt, n in lst if lo <= dt <= hi]
    if not names:
        k = "공시없음"
    else:
        k = kind_of(names)

    if code not in cache.get("_df", {}):
        pass
    try:
        df = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        continue
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Volume"] > 0]
    days = [int(x.strftime("%Y%m%d")) for x in df.index]
    if day not in days:
        continue
    i = days.index(day)
    n = len(df)
    if i + 2 >= n:
        continue
    o = df["Open"].values.astype(float)
    c = df["Close"].values.astype(float); h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    ma20 = pd.Series(c).rolling(20).mean().values
    pk = c[i]
    done = False
    for j in range(i + 2, min(i + 21, n - 2)):
        if c[j] > pk * 1.05:
            pk = c[j]
        dd = (c[j] / pk - 1) * 100
        near20 = ma20[j] > 0 and abs(c[j] / ma20[j] - 1) <= 0.05
        rebound = c[j] > c[j - 1]
        if dd <= -8 and (near20 or dd <= -15) and rebound:
            r = sim(o, h, l, c, j + 1, 0.10, 0.08, 15)
            if r is not None:
                rows.append(dict(kind=k, net=r * 100, code=code, day=day))
            done = True
            break
    if not done:
        continue

d = pd.DataFrame(rows)
print(f"[INFO] 표본 {len(d):,}건\n")
print(f"{'재료 성격':<12}{'건수':>7}{'비중':>7}{'승률':>7}{'건당%':>8}{'합계%':>9}")
for k in ("실적", "계약", "자본변동", "기타/테마", "공시없음"):
    x = d[d.kind == k].net.values
    if not len(x):
        continue
    print(f"{k:<12}{len(x):>7,}{len(x)/len(d)*100:>6.0f}%{(x>0).mean()*100:>7.1f}%"
          f"{x.mean():>+8.3f}{x.sum():>+9.0f}")
print(f"{'전체':<12}{len(d):>7,}{100:>6.0f}%{(d.net>0).mean()*100:>7.1f}%"
      f"{d.net.mean():>+8.3f}{d.net.sum():>+9.0f}")
