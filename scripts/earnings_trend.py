"""실적 공시 동반 급등은 추세가 이어지는가 (2년)

재료일(하루 +10% & 거래량 5배) 다음날 시가 매수 → 보유 기간별 성과.
공시 성격(실적 / 기타·테마 / 없음)으로 나눠 비교.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
FEE = 0.00015
EARN = ("분기보고서", "반기보고서", "사업보고서", "영업(잠정)실적", "잠정실적",
        "매출액또는손익구조", "실적공시", "연결재무제표")
DEAL = ("단일판매", "공급계약", "수주", "판매ㆍ공급계약")


def kind_of(names):
    txt = " ".join(names)
    if any(k in txt for k in EARN):
        return "실적"
    if any(k in txt for k in DEAL):
        return "계약"
    return "기타/테마"


def hold_ret(o, c, e, maxd):
    """다음날 시가 매수 후 maxd거래일 보유 수익률"""
    if e >= len(c):
        return None
    px = float(o[e])
    if px <= 0:
        return None
    j = min(e + maxd - 1, len(c) - 1)
    return (float(c[j]) / px - 1 - FEE) * 100


sigs = json.loads((ROOT / "data" / "surge_signals.json").read_text(encoding="utf-8"))
disc = json.loads((ROOT / "data" / "dart_disclosures.json").read_text(encoding="utf-8"))
by_code = {c: [(r["date"], r["name"]) for r in rows] for c, rows in disc.items()}

rows = []
for s in sigs:
    code, day = s["code"], s["day"]
    d0 = datetime.strptime(str(day), "%Y%m%d")
    lo = (d0 - timedelta(days=7)).strftime("%Y%m%d")
    hi = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    names = [n for dt, n in by_code.get(code, []) if lo <= dt <= hi]
    k = kind_of(names) if names else "공시없음"
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
    o = df["Open"].values.astype(float)
    c = df["Close"].values.astype(float)
    rec = dict(kind=k)
    for hd in (5, 10, 20, 40):
        rec[f"h{hd}"] = hold_ret(o, c, i + 1, hd)
    rows.append(rec)

d = pd.DataFrame(rows)
print(f"[INFO] 표본 {len(d):,}건\n")
print(f"{'재료 성격':<12}{'건수':>7}" + "".join(f"{f'{h}일':>9}" for h in (5, 10, 20, 40)))
for k in ("실적", "계약", "기타/테마", "공시없음"):
    sub = d[d.kind == k]
    if not len(sub):
        continue
    print(f"{k:<12}{len(sub):>7,}" + "".join(
        f"{sub[f'h{h}'].mean():>+9.2f}" for h in (5, 10, 20, 40)))
print(f"{'전체':<12}{len(d):>7,}" + "".join(
    f"{d[f'h{h}'].mean():>+9.2f}" for h in (5, 10, 20, 40)))
print("\n(단위 %, 수수료 반영, 손절 없이 보유)")
print(f"\n{'재료 성격':<12}{'20일 승률':>10}{'40일 승률':>10}")
for k in ("실적", "계약", "기타/테마", "공시없음"):
    sub = d[d.kind == k]
    if not len(sub):
        continue
    print(f"{k:<12}{(sub.h20>0).mean()*100:>9.1f}%{(sub.h40>0).mean()*100:>9.1f}%")
