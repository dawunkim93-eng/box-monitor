"""목표 재정의: 3일 내 탈출 + 시간손절 → 조건 재탐색 (6개월)

거래 규칙:
- 신호일 종가 = 지정가, 다음날 저가가 닿으면 체결
- 사다리 1~3차 (1.5% 간격, 6·6·9만원 = 21만), 각 차수 ×1.015 익절
- 시간손절: 진입일 종가가 3차 가격(진입가 -3%) 아래면 그날 종가 전량 청산
- 최대 3거래일 보유 후 남은 물량은 종가 청산

성공 정의: 순손익 > 0 (수익으로 끝난 거래)
탐색: 단일 조건 → 2개 조합 (성공률 리프트)
"""
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
OUT = ROOT / "data" / "daily_find3.json"
UNIV = json.loads((ROOT / "data/minute_universe.json").read_text())["codes"]
GAP, FEE = 0.015, 0.00015
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_N = 200
QS = (0.2, 0.4, 0.6, 0.8)

FIELDS = [("atr", "ATR"), ("width", "박스폭"), ("pos", "밴드위치"), ("rsi", "RSI"),
          ("amt", "거래대금"), ("ma_gap20", "20일선이격"), ("ma_gap60", "60일선이격"),
          ("ret3", "3일수익"), ("ret5", "5일수익"), ("ret10", "10일수익"),
          ("vol20", "변동성"), ("vratio", "거래량비"), ("pos52", "52주위치"),
          ("gap", "평균갭"), ("cl_pos", "종가위치")]
UNIT = dict(atr="%", width="%", pos="%", rsi="", amt="억", ma_gap20="%", ma_gap60="%",
            ret3="%", ret5="%", ret10="%", vol20="%", vratio="배", pos52="%",
            gap="%", cl_pos="%")


def build(code, start):
    try:
        df = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        return []
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Volume"] > 0]
    if len(df) < 300:
        return []
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c)
    tr = np.concatenate([[h[0] - l[0]],
                         np.maximum(h[1:] - l[1:],
                                    np.maximum(np.abs(h[1:] - c[:-1]),
                                               np.abs(l[1:] - c[:-1])))])
    atr = pd.Series(tr).ewm(alpha=1 / 14, min_periods=14).mean().values / np.maximum(c, 1e-9)
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / 14, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / 14, adjust=False).mean().values
    rsi = 100 - 100 / (1 + up / np.maximum(dn, 1e-9))
    cs = pd.Series(c)
    lo30 = cs.rolling(30).min().values
    hi30 = cs.rolling(30).max().values
    ma20 = cs.rolling(20).mean().values
    ma60 = cs.rolling(60).mean().values
    amt20 = pd.Series(c * v).rolling(20).mean().values
    vol20 = pd.Series(d / np.maximum(np.roll(c, 1), 1e-9)).rolling(20).std().values * 100
    vavg = pd.Series(v).rolling(20).mean().values
    gap20 = pd.Series(o / np.maximum(np.roll(c, 1), 1e-9) - 1).rolling(20).mean().values * 100
    lo52 = cs.rolling(250, min_periods=60).min().values
    hi52 = cs.rolling(250, min_periods=60).max().values

    out = []
    for i in range(260, n - 2):
        if df.index[i] < start:
            continue
        if lo30[i] <= 0 or hi30[i] <= lo30[i] or ma60[i] <= 0 or vavg[i] <= 0:
            continue
        limit = c[i]
        if l[i + 1] > limit:
            continue
        lv = [limit * (1 - GAP) ** k for k in range(3)]
        tgt = [x * (1 + GAP) for x in lv]
        held = [True, False, False]
        net = 0.0
        closed = False
        for dd in range(HOLD):
            j = i + 1 + dd
            if j >= n:
                break
            for k in range(1, 3):
                if not held[k] and l[j] <= lv[k]:
                    held[k] = True
            for k in range(3):
                if held[k] and h[j] >= tgt[k]:
                    held[k] = False
                    net += AMTS[k] * 10_000 * (GAP - FEE)
            if dd == 0 and any(held) and c[j] < lv[2]:
                for k in range(3):
                    if held[k]:
                        net += AMTS[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
                        held[k] = False
                closed = True
                break
            if not any(held):
                closed = True
                break
        if not closed and any(held):
            j = min(i + HOLD, n - 1)
            for k in range(3):
                if held[k]:
                    net += AMTS[k] * 10_000 * (c[j] / lv[k] - 1 - FEE)
        px = c[i]
        out.append(dict(
            code=code, day=int(df.index[i].strftime("%Y%m%d")),
            net=net / SLOT * 100, win=net > 0,
            atr=atr[i] * 100, width=(hi30[i] / lo30[i] - 1) * 100,
            pos=(px - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100,
            rsi=rsi[i], amt=amt20[i] / 1e8,
            ma_gap20=(px / ma20[i] - 1) * 100, ma_gap60=(px / ma60[i] - 1) * 100,
            ret3=(px / c[i - 3] - 1) * 100, ret5=(px / c[i - 5] - 1) * 100,
            ret10=(px / c[i - 10] - 1) * 100, vol20=vol20[i],
            vratio=v[i] / vavg[i], gap=gap20[i],
            pos52=((px - lo52[i]) / max(hi52[i] - lo52[i], 1e-9) * 100
                   if not np.isnan(lo52[i]) else np.nan),
            cl_pos=(c[i] - l[i]) / max(h[i] - l[i], 1e-9) * 100))
    return out


def ev(d, mask, base):
    sel = d[mask]
    if len(sel) < MIN_N:
        return None
    return dict(n=int(len(sel)), win=round(float(sel["win"].mean()) * 100, 1),
                net=round(float(sel["net"].mean()), 3),
                lift=round(float(sel["win"].mean()) * 100 / max(base, 0.1), 2))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    start = pd.Timestamp.now().normalize() - pd.DateOffset(months=args.months)
    rows = []
    for k, code in enumerate(UNIV, 1):
        rows += build(code, start)
        if k % 60 == 0:
            print(f"  {k}/{len(UNIV)} · 거래 {len(rows):,}", flush=True)
    d = pd.DataFrame(rows)
    if not len(d):
        print("[WARN] 표본 없음")
        return
    base = float(d["win"].mean()) * 100
    print(f"[INFO] 거래 {len(d):,}건 · 승률 {base:.1f}% · 평균 {d['net'].mean():+.3f}% · "
          f"최악 {d['net'].min():+.1f}%")

    singles = []
    for key, ko in FIELDS:
        col = d[key].replace([np.inf, -np.inf], np.nan)
        if col.notna().sum() < 500:
            continue
        for q in QS:
            cut = float(col.quantile(q))
            for op in (">=", "<="):
                mask = ((col >= cut) if op == ">=" else (col <= cut)).fillna(False)
                r = ev(d, mask, base)
                if r:
                    r.update(key=key, op=op, cut=cut,
                             expr=f"{ko} {op} {cut:.2f}{UNIT.get(key,'')}")
                    singles.append(r)
    singles.sort(key=lambda x: (-x["net"], -x["win"]))
    seen, uniq = set(), []
    for r in singles:
        if r["expr"] in seen:
            continue
        seen.add(r["expr"])
        uniq.append(r)
    singles = uniq

    pool = [r for r in singles if r["net"] > 0][:10]
    pairs = []
    for a, b in itertools.combinations(pool, 2):
        ma = ((d[a["key"]] >= a["cut"]) if a["op"] == ">=" else (d[a["key"]] <= a["cut"]))
        mb = ((d[b["key"]] >= b["cut"]) if b["op"] == ">=" else (d[b["key"]] <= b["cut"]))
        r = ev(d, (ma & mb).fillna(False), base)
        if r:
            r.update(expr=f"{a['expr']} & {b['expr']}")
            pairs.append(r)
    pairs.sort(key=lambda x: (-x["net"], -x["win"]))
    triples = []
    for combo in itertools.combinations(pool[:6], 3):
        if len({c_["key"] for c_ in combo}) < 3:
            continue
        mask = None
        for c_ in combo:
            m = ((d[c_["key"]] >= c_["cut"]) if c_["op"] == ">=" else (d[c_["key"]] <= c_["cut"]))
            mask = m if mask is None else (mask & m)
        r = ev(d, mask.fillna(False), base)
        if r:
            r.update(expr=" & ".join(c_["expr"] for c_ in combo))
            triples.append(r)
    triples.sort(key=lambda x: (-x["net"], -x["win"]))

    keep = ("expr", "n", "win", "net", "lift")
    out = dict(
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        window=dict(start=int(d["day"].min()), end=int(d["day"].max()),
                    months=args.months,
                    trades=int(len(d)), stocks=int(d["code"].nunique())),
        base=dict(win=round(base, 1), net=round(float(d["net"].mean()), 3)),
        rule="전일 종가 지정가 · 3차(1.5%) · 3일 보유 · 진입일 종가가 3차 아래면 그날 청산",
        singles=[{k: r[k] for k in keep} for r in singles[:10]],
        pairs=[{k: r[k] for k in keep} for r in pairs[:10]],
        triples=[{k: r[k] for k in keep} for r in triples[:6]],
        note=("성공 = 순손익 > 0. 표본 6개월·전 종목 일봉(1.12배 과대). "
              "승률과 평균손익을 함께 볼 것."),
    )
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  단일:")
    for r in out["singles"][:5]:
        print(f"    {r['expr']:<24} n={r['n']:>5} 승률 {r['win']:>4.1f}% 평균 {r['net']:>+6.3f}%")
    print("  조합:")
    for r in out["pairs"][:5]:
        print(f"    {r['expr'][:44]:<44} n={r['n']:>5} 승률 {r['win']:>4.1f}% 평균 {r['net']:>+6.3f}%")


if __name__ == "__main__":
    main()
