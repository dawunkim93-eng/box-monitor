"""실적 개선 필터 검증 — 바닥 전환과 결합

실적 개선 판정(분기 단일 기준, YoY):
  최근 2개 분기 영업이익이 전년 동기보다 많고, 최근 분기가 흑자
  공시 제출 시점(3/31, 5/15, 8/14, 11/14)까지만 반영 → 미래 정보 차단

A) 바닥 전환 기본 조건 + 실적 개선
B) 재료 급등 후 조정 매수 + 실적 개선
"""
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
FEE = 0.00015
MONTHS = 24
# (연도, 분기) → 반영 가능 시점(MMDD, 같은 해 기준). Q4는 다음 해 3/31
AVAIL = {1: (5, 15), 2: (8, 14), 3: (11, 14), 4: (3, 31)}


def quarters(code, fin):
    """단일분기 영업이익/매출 (누적 차분)"""
    f = fin.get(code) or {}
    out = {}
    for y in (2024, 2025, 2026):
        q1 = f.get(f"{y}Q1") or {}
        q2 = f.get(f"{y}Q2") or {}
        q3 = f.get(f"{y}Q3") or {}
        q4 = f.get(f"{y}Q4") or {}
        v = {1: (q1.get("op"), q1.get("rev"))}
        if q2.get("op") is not None and q1.get("op") is not None:
            v[2] = (q2["op"] - q1["op"],
                    (q2["rev"] - q1["rev"]) if q2.get("rev") and q1.get("rev") else None)
        if q3.get("op") is not None and q2.get("op") is not None:
            v[3] = (q3["op"] - q2["op"],
                    (q3["rev"] - q2["rev"]) if q3.get("rev") and q2.get("rev") else None)
        if q4.get("op") is not None and q3.get("op") is not None:
            v[4] = (q4["op"] - q3["op"],
                    (q4["rev"] - q3["rev"]) if q4.get("rev") and q3.get("rev") else None)
        for q, t in v.items():
            if t[0] is not None:
                out[(y, q)] = t
    return out


def avail_qs(day):
    """day 시점에 공시된 (연도,분기) 목록 — 최신순"""
    y = day // 10000
    md = (day // 100) % 100 * 100 + day % 100
    lst = []
    for yy in (y, y - 1, y - 2):
        for q in (1, 2, 3, 4):
            ay = yy + 1 if q == 4 else yy
            am, ad = AVAIL[q]
            if (ay * 10000 + am * 100 + ad) <= day:
                lst.append((yy, q))
    return sorted(lst, reverse=True)


def improving(qv, day, need=2):
    """최근 need개 분기 연속 YoY 영업이익 증가 + 최근 분기 흑자"""
    if not qv:
        return False
    ok = 0
    for (y, q) in avail_qs(day):
        cur = qv.get((y, q))
        prev = qv.get((y - 1, q))
        if cur is None or prev is None or cur[0] is None or prev[0] is None:
            continue
        if cur[0] > 0 and cur[0] > prev[0]:
            ok += 1
            if ok >= need:
                return True
        else:
            return False
    return False


def sim(o, h, l, c, e, tgt=0.10, stop=0.10, maxd=20):
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


def main():
    fin = json.loads((ROOT / "data" / "dart_fin.json").read_text(encoding="utf-8"))
    codes = sorted(set(json.loads((ROOT / "data" / "daily_universe.json").read_text(encoding="utf-8"))["codes"]))
    start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
    A, B = [], []
    for code in codes:
        if code[-1] in "579K":
            continue
        try:
            df = pd.read_pickle(PIT / f"{code}.pkl")
        except Exception:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df["Volume"] > 0]
        if len(df) < 320:
            continue
        o = df["Open"].values.astype(float)
        c = df["Close"].values.astype(float); h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float); v = df["Volume"].values.astype(float)
        n = len(df)
        cs = pd.Series(c)
        ma5 = cs.rolling(5).mean().values
        ma20 = cs.rolling(20).mean().values
        v5 = pd.Series(v).rolling(5).mean().values
        v20 = pd.Series(v).rolling(20).mean().values
        lo60 = cs.rolling(60).min().values
        amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
        qv = quarters(code, fin)
        days = [int(x.strftime("%Y%m%d")) for x in df.index]
        for i in range(270, n - 25):
            if df.index[i] < start or amt20[i] < 10.0 or v20[i] <= 0:
                continue
            d0 = days[i]
            good = improving(qv, d0)
            # A 바닥 전환
            if (ma5[i] > ma20[i] and c[i] > ma20[i] and v5[i] / v20[i] >= 1.5
                    and any(ma5[k] > ma20[k] and ma5[k - 1] <= ma20[k - 1] for k in range(i - 4, i + 1))):
                up = (c[i] / lo60[i] - 1) * 100 if lo60[i] > 0 else 999
                if 5 < up <= 25:
                    r = sim(o, h, l, c, i + 1)
                    if r is not None:
                        A.append(dict(net=r * 100, good=good))
            # B 재료 급등 후 조정
            if c[i - 1] > 0 and (c[i] / c[i - 1] - 1) * 100 >= 10 and v[i] / v20[i] >= 5:
                pk = c[i]
                for j in range(i + 2, min(i + 21, n - 2)):
                    if c[j] > pk * 1.05:
                        pk = c[j]
                    dd = (c[j] / pk - 1) * 100
                    near20 = ma20[j] > 0 and abs(c[j] / ma20[j] - 1) <= 0.05
                    if dd <= -8 and (near20 or dd <= -15) and c[j] > c[j - 1]:
                        r = sim(o, h, l, c, j + 1, 0.10, 0.08, 15)
                        if r is not None:
                            B.append(dict(net=r * 100, good=good))
                        break

    for name, rows in (("A 바닥 전환", A), ("B 급등 후 조정", B)):
        d = pd.DataFrame(rows)
        print(f"\n[{name}] {'건수':>7}{'승률':>8}{'건당%':>9}{'합계%':>9}")
        for label, sub in (("전체", d), ("실적 개선 O", d[d.good]), ("실적 개선 X", d[~d.good])):
            if not len(sub):
                print(f"{label:<14}{0:>7}")
                continue
            x = sub.net.values
            print(f"{label:<14}{len(x):>7,}{(x>0).mean()*100:>7.1f}%{x.mean():>+9.3f}{x.sum():>+9.0f}")


if __name__ == "__main__":
    main()
