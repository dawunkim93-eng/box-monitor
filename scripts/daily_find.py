"""6개월 일봉으로 '전일 종가 지정가 → 당일 탈출' 신호 표본 생성 + 조건 탐색
→ data/daily_find.json

가정 (분봉 실측 대비 1.12배 과대 — 앞서 검증):
- 신호일 종가 = 지정가
- 체결: 다음날 저가 ≤ 지정가
- 당일 탈출: 다음날 고가 ≥ 지정가×1.015 (저가→고가 순서 가정)
- 5일 내 탈출: 이후 5거래일 내 고가가 목표 도달
- 순익: 슬롯 87만 대비 (1차 6만 + 체결된 차수만큼 +1.5%)

표본: 유니버스(거래대금 상위 300 + 박스 73) × 최근 6개월
출력: 신호 단위 데이터셋 요약 + 단일/2개/3개 조건 (리프트 = 당일 성공률 배수)
"""
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "minute_universe.json"
LIVE = ROOT / "data" / "box_live.json"
OUT = ROOT / "data" / "daily_find.json"

GAP = 0.015
AMTS = [6, 6, 9, 12, 18, 18, 18]
SLOT = sum(AMTS) * 10_000
FEE = 0.00015
MONTHS = 6
MIN_N = 200          # 신호 표본이 크므로 기준 상향
QS = (0.2, 0.4, 0.6, 0.8)


def universe():
    codes = []
    if UNIV.exists():
        codes += [str(c) for c in json.loads(UNIV.read_text(encoding="utf-8"))["codes"]]
    if LIVE.exists():
        for r in json.loads(LIVE.read_text(encoding="utf-8")).get("rows", []):
            c = str(r.get("code") or "")
            if len(c) == 6:
                codes.append(c)
    return sorted(set(codes))


def build_signals(code, start_ts):
    try:
        df = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        return []
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Volume"] > 0]
    if len(df) < 260:
        return []
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]),
                                              np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
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
    vavg20 = pd.Series(v).rolling(20).mean().values
    gap20 = pd.Series(o / np.maximum(np.roll(c, 1), 1e-9) - 1).rolling(20).mean().values * 100
    lo52 = pd.Series(c).rolling(250, min_periods=60).min().values
    hi52 = pd.Series(c).rolling(250, min_periods=60).max().values

    rows = []
    for i in range(260, n - 1):
        if df.index[i] < start_ts:
            continue
        px = c[i]
        if lo30[i] <= 0 or hi30[i] <= lo30[i] or ma60[i] <= 0 or vavg20[i] <= 0:
            continue
        limit = px
        tgt = limit * (1 + GAP)
        fill = l[i + 1] <= limit
        same = bool(fill and h[i + 1] >= tgt)
        esc5 = False
        esc_day = None
        net = 0.0
        if fill:
            for k in range(1, 5):
                j = i + 1 + k
                if j >= n:
                    break
                if h[j] >= tgt:
                    esc5 = True
                    esc_day = k + 1
                    break
            if same:
                esc5 = True
                esc_day = 1
            # 1차 + 당일 채워진 차수만 익절 가정 (간이)
            levels = [limit * (1 - GAP) ** k for k in range(7)]
            cycles = 1 if (same or esc5) else 0
            for k in range(1, 7):
                if l[i + 1] <= levels[k]:
                    cycles += 1
            net = cycles * AMTS[0] * 10_000 * (GAP - FEE)
        rows.append(dict(
            code=code, day=int(df.index[i].strftime("%Y%m%d")),
            fill=bool(fill), same=bool(same), esc5=bool(esc5), esc_day=esc_day,
            net=net,
            atr=atr[i] * 100, width=(hi30[i] / lo30[i] - 1) * 100,
            pos=(px - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100,
            rsi=rsi[i], amt=amt20[i] / 1e8,
            ma_gap20=(px / ma20[i] - 1) * 100, ma_gap60=(px / ma60[i] - 1) * 100,
            ret3=(px / c[i - 3] - 1) * 100, ret5=(px / c[i - 5] - 1) * 100,
            ret10=(px / c[i - 10] - 1) * 100, vol20=vol20[i],
            vratio=v[i] / vavg20[i], gap=gap20[i],
            pos52=((px - lo52[i]) / max(hi52[i] - lo52[i], 1e-9) * 100
                   if not np.isnan(lo52[i]) else np.nan),
            cl_pos=(c[i] - l[i]) / max(h[i] - l[i], 1e-9) * 100))
    return rows


FIELDS = [("atr", "ATR"), ("width", "박스폭"), ("pos", "밴드위치"), ("rsi", "RSI"),
          ("amt", "거래대금"), ("ma_gap20", "20일선이격"), ("ma_gap60", "60일선이격"),
          ("ret3", "3일수익"), ("ret5", "5일수익"), ("ret10", "10일수익"),
          ("vol20", "변동성"), ("vratio", "거래량비"), ("pos52", "52주위치"),
          ("gap", "평균갭"), ("cl_pos", "종가위치")]
UNIT = dict(atr="%", width="%", pos="%", rsi="", amt="억", ma_gap20="%", ma_gap60="%",
            ret3="%", ret5="%", ret10="%", vol20="%", vratio="배", pos52="%",
            gap="%", cl_pos="%")


def ev(d, mask, base):
    sel = d[mask]
    if len(sel) < MIN_N:
        return None
    filled = sel[sel["fill"]]
    return dict(n=int(len(sel)), rate=round(float((sel["fill"] & sel["same"]).mean()) * 100, 1),
                fill=round(float(sel["fill"].mean()) * 100, 1),
                same=round(float(filled["same"].mean()) * 100, 1) if len(filled) else 0.0,
                esc5=round(float(filled["esc5"].mean()) * 100, 1) if len(filled) else 0.0,
                lift=round(float((sel["fill"] & sel["same"]).mean()) * 100 / max(base, 0.1), 2))


def main():
    codes = universe()
    start = pd.Timestamp.now().normalize() - pd.DateOffset(months=MONTHS)
    rows = []
    for i, code in enumerate(codes, 1):
        rows += build_signals(code, start)
        if i % 50 == 0:
            print(f"  {i}/{len(codes)} · 신호 {len(rows):,}", flush=True)
    if not rows:
        print("[WARN] 신호 없음")
        return
    d = pd.DataFrame(rows)
    base = float((d["fill"] & d["same"]).mean()) * 100
    print(f"[INFO] 신호 {len(d):,}건 · 체결 {d['fill'].mean()*100:.1f}% · "
          f"당일성공 {base:.1f}% · 종목 {d['code'].nunique()} · 기간 {d['day'].min()}~{d['day'].max()}")

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
                if not r:
                    continue
                r.update(key=key, op=op, cut=cut, expr=f"{ko} {op} {cut:.2f}{UNIT.get(key,'')}")
                singles.append(r)
    singles.sort(key=lambda x: (-x["rate"], -x["same"]))
    seen, uniq = set(), []
    for r in singles:
        if r["expr"] in seen:
            continue
        seen.add(r["expr"])
        uniq.append(r)
    singles = uniq

    pool = [r for r in singles if r["lift"] >= 1.15][:10]
    pairs, triples = [], []
    for a, b in itertools.combinations(pool, 2):
        ma = ((d[a["key"]] >= a["cut"]) if a["op"] == ">=" else (d[a["key"]] <= a["cut"]))
        mb = ((d[b["key"]] >= b["cut"]) if b["op"] == ">=" else (d[b["key"]] <= b["cut"]))
        r = ev(d, (ma & mb).fillna(False), base)
        if r:
            r.update(expr=f"{a['expr']} & {b['expr']}")
            pairs.append(r)
    pairs.sort(key=lambda x: (-x["rate"], -x["same"]))
    for combo in itertools.combinations(pool[:6], 3):
        if len({c["key"] for c in combo}) < 3:
            continue
        mask = None
        for c_ in combo:
            m = ((d[c_["key"]] >= c_["cut"]) if c_["op"] == ">=" else (d[c_["key"]] <= c_["cut"]))
            mask = m if mask is None else (mask & m)
        r = ev(d, mask.fillna(False), base)
        if r:
            r.update(expr=" & ".join(c_["expr"] for c_ in combo))
            triples.append(r)
    triples.sort(key=lambda x: (-x["rate"], -x["same"]))

    keep = ("expr", "n", "rate", "fill", "same", "esc5", "lift")
    out = dict(
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        window=dict(start=int(d["day"].min()), end=int(d["day"].max()),
                    signals=int(len(d)), stocks=int(d["code"].nunique())),
        base=dict(rate=round(base, 1), fill=round(float(d["fill"].mean()) * 100, 1)),
        singles=[{k: r[k] for k in keep} for r in singles[:10]],
        pairs=[{k: r[k] for k in keep} for r in pairs[:10]],
        triples=[{k: r[k] for k in keep} for r in triples[:8]],
        note=("일봉 근사(저가 체결·고가 목표) — 분봉 실측 대비 사이클 1.12배 과대. "
              "당일성공 = 체결되고 그날 +1.5%에 팔린 비율. 리프트 = 전체 대비 배수."),
    )
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  단일:")
    for r in out["singles"][:5]:
        print(f"    {r['expr']:<24} n={r['n']:>5} 당일 {r['rate']:>4.1f}% 체결 {r['fill']:>4.1f}% L{r['lift']}")
    print("  조합:")
    for r in out["pairs"][:5]:
        print(f"    {r['expr'][:46]:<46} n={r['n']:>5} 당일 {r['rate']:>4.1f}% L{r['lift']}")


if __name__ == "__main__":
    main()
