"""분봉 성과 ↔ 일봉 특징 대조 + 조건 조합 탐색 → data/minute_find.json

입력: data/minute_review.json (종목별 신호당 기대수익)
특징(17종): ATR, 6주 밴드 폭, 밴드 위치, 일봉 RSI, 거래대금, 20/60일선 이격,
             3/5/10일 수익률, 20일 변동성, 볼린저 폭, 거래량비, 52주 위치,
             연속 상승일, 평균 갭, 당일 종가 위치
성공: 신호당 기대수익 상위 30%
탐색: 단일 조건(분위 3구간 × 양방향) → 상위 조건끼리 2개 조합(AND)

⚠ 표본이 6~10거래일이면 우연이 섞임. 리프트와 표본 수를 함께 볼 것.
"""
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PIT = Path("/home/lucky/backtest/data/pit")
REV = ROOT / "data" / "minute_review.json"
OUT = ROOT / "data" / "minute_find.json"
WEEKS = 6
TOP_Q = 0.30
MIN_N = 25
QS = (0.25, 0.5, 0.75)


def features(code):
    try:
        df = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Volume"] > 0]
    if len(df) < 80:
        return None
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float)
    px = c[-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]),
                                              np.abs(l[1:] - c[:-1])))
    atr = float(pd.Series(tr).ewm(alpha=1 / 14, min_periods=14).mean().iloc[-1]) / px * 100
    wk = pd.Series(c, index=df.index).resample("W-FRI").last().dropna()
    if len(wk) < WEEKS:
        return None
    seg = wk.values[-WEEKS:].astype(float)
    lo, hi = seg.min(), seg.max()
    width = (hi / lo - 1) * 100
    pos = (px - lo) / max(hi - lo, 1e-9) * 100
    d = np.diff(c[-15:])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = float((100 - 100 / (1 + up / max(dn.iloc[-1], 1e-9))).iloc[-1])
    amt20 = float((c[-20:] * v[-20:]).mean()) / 1e8
    ma20 = float(c[-20:].mean())
    ma60 = float(c[-60:].mean()) if len(c) >= 60 else ma20
    vol20 = float(np.std(np.diff(c[-21:]) / c[-21:-1])) * 100
    sd20 = float(np.std(c[-20:]))
    lo52 = float(c[-250:].min()) if len(c) >= 250 else float(c.min())
    hi52 = float(c[-250:].max()) if len(c) >= 250 else float(c.max())
    up_days = 0
    for k in range(len(c) - 1, 0, -1):
        if c[k] > c[k - 1]:
            up_days += 1
        else:
            break
    return dict(
        atr=atr, width=width, pos=pos, rsi=rsi, amt=amt20,
        ma_gap20=(px / ma20 - 1) * 100, ma_gap60=(px / ma60 - 1) * 100,
        ret3=(px / c[-4] - 1) * 100, ret5=(px / c[-6] - 1) * 100,
        ret10=(px / c[-11] - 1) * 100, vol20=vol20,
        bb=(4 * sd20) / ma20 * 100,
        vratio=float(v[-1] / max(v[-20:].mean(), 1e-9)),
        pos52=(px - lo52) / max(hi52 - lo52, 1e-9) * 100,
        up_days=float(up_days),
        gap=float(np.mean((o[-20:] / c[-21:-1] - 1) * 100)),
        cl_pos=(c[-1] - l[-1]) / max(h[-1] - l[-1], 1e-9) * 100)


FIELDS = [("atr", "ATR"), ("width", "박스폭"), ("pos", "밴드위치"), ("rsi", "RSI"),
          ("amt", "거래대금"), ("ma_gap20", "20일선이격"), ("ma_gap60", "60일선이격"),
          ("ret3", "3일수익"), ("ret5", "5일수익"), ("ret10", "10일수익"),
          ("vol20", "변동성"), ("bb", "볼린저폭"), ("vratio", "거래량비"),
          ("pos52", "52주위치"), ("up_days", "연속상승"), ("gap", "평균갭"),
          ("cl_pos", "종가위치")]

UNIT = dict(atr="%", width="%", pos="%", rsi="", amt="억", ma_gap20="%",
            ma_gap60="%", ret3="%", ret5="%", ret10="%", vol20="%", bb="%",
            vratio="배", pos52="%", up_days="일", gap="%", cl_pos="%")


def evaluate(df, mask):
    sel = df[mask]
    if len(sel) < MIN_N:
        return None
    return dict(n=int(len(sel)), ret=round(float(sel["ret"].mean()), 3),
                top=round(float(sel["top"].mean()) * 100, 1),
                fill=round(float(sel["fill"].mean()), 1),
                esc=round(float(sel["esc"].mean()), 1),
                same=round(float(sel["same"].mean()), 1),
                days=round(float(sel["days"].mean()), 2),
                score=round(float(sel["fill"].mean()) * float(sel["same"].mean()) / 100, 1))


def main():
    if not REV.exists():
        print("[WARN] minute_review.json 없음")
        return
    rev = json.loads(REV.read_text(encoding="utf-8"))
    rows = []
    for s in rev.get("stocks", []):
        f = features(s["code"])
        if not f:
            continue
        rows.append(dict(code=s["code"], name=s["name"], ret=s["ret_sig"],
                         fill=s["fill"], esc=s["esc"], same=s["same"],
                         days=s["days"], **f))
    if len(rows) < 40:
        print(f"[WARN] 표본 부족 ({len(rows)}종목)")
        return

    d = pd.DataFrame(rows)
    thr = float(d["ret"].quantile(1 - TOP_Q))
    d["top"] = d["ret"] >= thr
    base_rate = float(d["top"].mean()) * 100

    singles = []
    for key, ko in FIELDS:
        col = d[key].replace([np.inf, -np.inf], np.nan)
        if col.notna().sum() < 40:
            continue
        for q in QS:
            cut = float(col.quantile(q))
            for op in (">=", "<="):
                mask = ((col >= cut) if op == ">=" else (col <= cut)).fillna(False)
                st = evaluate(d, mask)
                if not st:
                    continue
                st.update(key=key, op=op, cut=cut,
                          expr=f"{ko} {op} {cut:.2f}{UNIT.get(key, '')}",
                          lift=round(st["top"] / max(base_rate, 0.1), 2))
                singles.append(st)

    seen, uniq = set(), []
    for r in sorted(singles, key=lambda x: (-x["score"], -x["same"], x["days"])):
        if r["expr"] in seen:
            continue
        seen.add(r["expr"])
        uniq.append(r)
    singles = uniq

    pool = [r for r in singles if r["lift"] >= 1.1][:12]
    pairs = []
    for a, b in itertools.combinations(pool, 2):
        if a["key"] == b["key"]:
            continue
        ma = ((d[a["key"]] >= a["cut"]) if a["op"] == ">=" else (d[a["key"]] <= a["cut"]))
        mb = ((d[b["key"]] >= b["cut"]) if b["op"] == ">=" else (d[b["key"]] <= b["cut"]))
        st = evaluate(d, (ma & mb).fillna(False))
        if not st:
            continue
        st.update(expr=f"{a['expr']} & {b['expr']}",
                  lift=round(st["top"] / max(base_rate, 0.1), 2))
        pairs.append(st)
    pairs.sort(key=lambda x: (-x["score"], -x["same"], x["days"]))

    tri_pool = [r for r in singles if r["lift"] >= 1.2][:6]
    triples = []
    for combo in itertools.combinations(tri_pool, 3):
        if len({c["key"] for c in combo}) < 3:
            continue
        mask = None
        for c in combo:
            m = ((d[c["key"]] >= c["cut"]) if c["op"] == ">=" else (d[c["key"]] <= c["cut"]))
            mask = m if mask is None else (mask & m)
        st = evaluate(d, mask.fillna(False))
        if not st:
            continue
        st.update(expr=" & ".join(c["expr"] for c in combo),
                  lift=round(st["top"] / max(base_rate, 0.1), 2))
        triples.append(st)
    triples.sort(key=lambda x: (-x["score"], -x["same"], x["days"]))

    feats = []
    for key, ko in FIELDS:
        col = d[key].replace([np.inf, -np.inf], np.nan)
        if col.notna().sum() < 40:
            continue
        try:
            qs = pd.qcut(col, 4, duplicates="drop")
        except ValueError:
            continue
        buckets = []
        for _, idx in col.groupby(qs, observed=True).groups.items():
            seg = d.loc[idx]
            buckets.append(dict(label=f"{seg[key].min():.2f}~{seg[key].max():.2f}",
                                n=int(len(seg)),
                                top=round(float(seg["top"].mean()) * 100, 1),
                                ret=round(float(seg["ret"].mean()), 3)))
        feats.append(dict(key=key, ko=ko, unit=UNIT.get(key, ""), buckets=buckets))

    out = dict(
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        window=rev.get("window"),
        base=dict(stocks=len(d), top_q=TOP_Q * 100, top_rate=round(base_rate, 1),
                  top_thr=round(thr, 3), avg_ret=round(float(d["ret"].mean()), 3)),
        singles=[{k: r[k] for k in ("expr", "n", "ret", "top", "lift", "fill", "esc", "same", "days", "score")}
                 for r in singles[:12]],
        pairs=[{k: r[k] for k in ("expr", "n", "ret", "top", "lift", "fill", "esc", "same", "days", "score")}
               for r in pairs[:10]],
        triples=[{k: r[k] for k in ("expr", "n", "ret", "top", "lift", "fill", "esc", "same", "days", "score")}
                 for r in triples[:6]],
        features=feats,
        note=("성공 = 종목별 신호당 기대수익 상위 30%. 적중률 = 걸러낸 종목 중 성공 비율, "
              "리프트 = 전체 성공률 대비 배수. 조건은 마지막 수집일 기준 일봉 특징. "
              "표본 6~10거래일이면 우연이 섞이므로 리프트 1.3 이상·표본 30건 이상만 신뢰. "
              "체결율 = 지정가까지 내려와 실제로 산 비율, 탈출율 = 산 것 중 1차가 +1.5%에 팔린 비율. "
              "종합 = 체결율 × 당일 탈출률 (신호 1건이 당일 탈출로 이어질 확률)."),
    )
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[INFO] {len(d)}종목 · 성공률 {base_rate:.1f}% (기준 {thr:.3f}%)")
    print("  단일:")
    for r in out["singles"][:6]:
        print(f"    {r['expr']:<26} n={r['n']:>3} 평균 {r['ret']:>6.3f}% 적중 {r['top']:>5.1f}% L{r['lift']}")
    print("  3개 조합 (종합순):")
    for r in out["triples"][:3]:
        print(f"    {r['expr'][:56]:<56} n={r['n']:>3} 종합 {r['score']:>5.1f} 체결 {r['fill']:>4.1f}% 당일 {r['same']:>4.1f}% L{r['lift']}")
    print("  조합:")
    for r in out["pairs"][:6]:
        print(f"    {r['expr'][:50]:<50} n={r['n']:>3} 종합 {r['score']:>5.1f} 체결 {r['fill']:>4.1f}% 당일 {r['same']:>4.1f}% L{r['lift']}")


if __name__ == "__main__":
    main()
