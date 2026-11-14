"""분봉 성과 ↔ 일봉 특징 대조로 조건 후보 발굴 → data/minute_find.json

입력: data/minute_review.json (종목별 신호당 기대수익)
특징: 일봉 PIT에서 마지막 수집일 기준으로 계산
  ATR14(%) · 6주 종가밴드 폭(%) · 밴드 내 위치(%) · 일봉 RSI14 ·
  20일 평균 거래대금(억) · 20일선 이격(%) · 5일 수익률(%) · 20일 변동성(%)
성공 정의: 신호당 기대수익 상위 30% (또는 0 초과면 상위)
출력: 조건 후보별 종목 수·평균 수익·상위군 적중률·리프트
"""
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


def features(code):
    try:
        df = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Volume"] > 0]
    if len(df) < 60:
        return None
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    px = c[-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]),
                                              np.abs(l[1:] - c[:-1])))
    atr = float(pd.Series(tr).ewm(alpha=1 / 14, min_periods=14).mean().iloc[-1]) / px
    wk = pd.Series(c, index=df.index).resample("W-FRI").last().dropna()
    if len(wk) < WEEKS:
        return None
    seg = wk.values[-WEEKS:].astype(float)
    lo, hi = seg.min(), seg.max()
    width = hi / lo - 1
    pos = (px - lo) / max(hi - lo, 1e-9)
    d = np.diff(c[-15:])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = float((100 - 100 / (1 + up / max(dn.iloc[-1], 1e-9))).iloc[-1])
    amt20 = float((c[-20:] * v[-20:]).mean()) / 1e8
    ma20 = float(c[-20:].mean())
    ma_gap = px / ma20 - 1
    ret5 = px / c[-6] - 1
    vol20 = float(np.std(np.diff(c[-21:]) / c[-21:-1])) * 100
    return dict(atr=atr * 100, width=width * 100, pos=pos * 100, rsi=rsi,
                amt=amt20, ma_gap=ma_gap * 100, ret5=ret5 * 100, vol=vol20)


FIELDS = [("atr", "ATR14(%)"), ("width", "박스 폭(%)"), ("pos", "밴드 위치(%)"),
          ("rsi", "일봉 RSI"), ("amt", "거래대금(억)"), ("ma_gap", "20일선 이격(%)"),
          ("ret5", "5일 수익률(%)"), ("vol", "20일 변동성(%)")]


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
                         fill=s["fill"], esc=s["esc"], pos_n=s["pos"], **f))
    if len(rows) < 30:
        print(f"[WARN] 표본 부족 ({len(rows)}종목)")
        return

    d = pd.DataFrame(rows)
    thr = d["ret"].quantile(1 - TOP_Q)
    d["top"] = d["ret"] >= thr
    base_rate = d["top"].mean() * 100
    base_ret = d["ret"].mean()

    feats, rules = [], []
    for key, ko in FIELDS:
        col = d[key].replace([np.inf, -np.inf], np.nan).dropna()
        if len(col) < 30:
            continue
        try:
            qs = pd.qcut(col, 4, duplicates="drop")
        except ValueError:
            continue
        buckets = []
        for lab, idx in col.groupby(qs, observed=True).groups.items():
            seg = d.loc[idx]
            buckets.append(dict(label=f"{seg[key].min():.1f}~{seg[key].max():.1f}",
                                n=int(len(seg)),
                                top=round(seg["top"].mean() * 100, 1),
                                ret=round(seg["ret"].mean(), 3)))
        feats.append(dict(key=key, ko=ko, buckets=buckets))

        # 단순 임계 규칙
        for th in [0.25, 0.5, 0.75]:
            cut = col.quantile(th)
            sel = d[d[key] >= cut] if th == 0.5 else d[d[key] >= col.quantile(0.5)]
            if th != 0.5:
                continue
            if len(sel) < 20:
                continue
            rules.append(dict(expr=f"{ko} ≥ {cut:.1f}", n=int(len(sel)),
                              ret=round(sel["ret"].mean(), 3),
                              top=round(sel["top"].mean() * 100, 1),
                              lift=round(sel["top"].mean() * 100 / max(base_rate, 0.1), 2)))
            sel2 = d[d[key] <= cut]
            if len(sel2) < 20:
                continue
            rules.append(dict(expr=f"{ko} ≤ {cut:.1f}", n=int(len(sel2)),
                              ret=round(sel2["ret"].mean(), 3),
                              top=round(sel2["top"].mean() * 100, 1),
                              lift=round(sel2["top"].mean() * 100 / max(base_rate, 0.1), 2)))

    rules.sort(key=lambda x: (-x["lift"], -x["ret"]))
    top_rules = [r for r in rules if r["n"] >= 30][:8]

    out = dict(
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        window=rev.get("window"),
        base=dict(stocks=len(d), top_q=TOP_Q * 100, top_rate=round(base_rate, 1),
                  top_thr=round(float(thr), 3), avg_ret=round(float(base_ret), 3)),
        features=feats, rules=top_rules,
        note=("성공 정의: 종목별 신호당 기대수익 상위 30%. 적중률 = 선택한 종목 중 성공 비율, "
              "리프트 = 전체 성공률 대비 배수. 표본이 6거래일뿐이라 탐색용으로만 볼 것."),
    )
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[INFO] {len(d)}종목 · 성공률 {base_rate:.1f}% (상위 {TOP_Q*100:.0f}%, 기준 {thr:.3f}%)")
    for r in top_rules:
        print(f"  {r['expr']:<22} n={r['n']:>3} · 평균 {r['ret']:>7.3f}% · 적중 {r['top']:>5.1f}% · 리프트 {r['lift']}")


if __name__ == "__main__":
    main()
