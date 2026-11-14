"""저변동 모의매매 — 매일 후보 생성 + 결과 자동 채점

조건 (2년 검증 통과):
  A: 거래대금(20일) ≤ 37.5억 & 밴드위치(30일 종가밴드) ≥ 89.5%
  B: 거래대금(20일) ≤ 37.5억 & ATR ≤ 5.85%

거래 규칙:
  - 신호일 종가 = 지정가, 다음날 저가가 닿으면 체결
  - 3차 사다리: 지정가, -1.5%, -3.0% (6·6·9만원 = 슬롯 21만원)
  - 각 차수 +1.5% 터치 시 매도
  - 시간손절: 진입일 종가가 3차 가격 아래면 그날 종가 전량 청산
  - 최대 3거래일 보유 후 남은 물량 종가 청산

산출:
  data/paper_picks.json  — 누적 후보 (code, sig, limit, cond)
  data/daily_pick.json   — 오늘의 후보 + 성과 요약 + 최근 내역
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
PICKS = ROOT / "data" / "paper_picks.json"
OUT = ROOT / "data" / "daily_pick.json"

GAP, FEE = 0.015, 0.00015
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_HIST = 60
MIN_AMT = 10.0      # 실전 최소 유동성(20일 평균 거래대금, 억원)


def load(code):
    try:
        df = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Volume"] > 0]
    return df if len(df) >= MIN_HIST else None


def features(df):
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c)
    tr = np.concatenate([[h[0] - l[0]],
                         np.maximum(h[1:] - l[1:],
                                    np.maximum(np.abs(h[1:] - c[:-1]),
                                               np.abs(l[1:] - c[:-1])))])
    atr = pd.Series(tr).ewm(alpha=1 / 14, min_periods=14).mean().values / np.maximum(c, 1e-9)
    cs = pd.Series(c)
    lo30 = cs.rolling(30).min().values
    hi30 = cs.rolling(30).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    i = n - 1
    if lo30[i] <= 0 or hi30[i] <= lo30[i]:
        return None
    return dict(close=float(c[i]), atr=float(atr[i] * 100), amt=float(amt20[i]),
                pos=float((c[i] - lo30[i]) / (hi30[i] - lo30[i]) * 100))


def evaluate(code, df, sig, limit):
    """신호 이후 결과 계산. dict 또는 None(미체결)."""
    days = [int(x.strftime("%Y%m%d")) for x in df.index]
    if sig not in days:
        return None
    i = days.index(sig)
    if i + 1 >= len(df):
        return None
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    if float(l[i + 1]) > limit:
        return dict(status="미체결", net=0.0, days=0, entry=None, exit=None)
    lv = [limit * (1 - GAP) ** k for k in range(3)]
    tg = [x * (1 + GAP) for x in lv]
    held = [True, False, False]
    net = 0.0
    exit_kind = None
    for dd in range(HOLD):
        j = i + 1 + dd
        if j >= len(df):
            break
        for k in range(1, 3):
            if not held[k] and float(l[j]) <= lv[k]:
                held[k] = True
        for k in range(3):
            if held[k] and float(h[j]) >= tg[k]:
                held[k] = False
                net += AMTS[k] * 10_000 * (GAP - FEE)
        if dd == 0 and any(held) and float(c[j]) < lv[2]:
            for k in range(3):
                if held[k]:
                    net += AMTS[k] * 10_000 * (float(c[j]) / lv[k] - 1 - FEE)
                    held[k] = False
            exit_kind = "시간손절"
            break
        if not any(held):
            exit_kind = exit_kind or "전량익절"
            break
    if any(held):
        j = min(i + HOLD, len(df) - 1)
        for k in range(3):
            if held[k]:
                net += AMTS[k] * 10_000 * (float(c[j]) / lv[k] - 1 - FEE)
        exit_kind = exit_kind or "기간청산"
    exit_day = int(df.index[min(i + HOLD, len(df) - 1)].strftime("%Y%m%d"))
    return dict(status="청산완료", net=net / SLOT * 100, days=min(HOLD, len(df) - 1 - i),
                entry=int(df.index[i + 1].strftime("%Y%m%d")),
                exit=exit_day, kind=exit_kind)


def main():
    codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
    master = pd.read_pickle(PIT / "master.pkl")
    names = dict(zip(master["code"].astype(str), master["name"].astype(str)))

    picks = json.loads(PICKS.read_text(encoding="utf-8")) if PICKS.exists() else []
    seen = {(p["code"], p["sig"]) for p in picks}
    today = None
    added = []
    for code in codes:
        df = load(code)
        if df is None:
            continue
        f = features(df)
        if not f:
            continue
        sig = int(df.index[-1].strftime("%Y%m%d"))
        today = sig
        cond = []
        if f["amt"] >= MIN_AMT and f["amt"] <= 37.51 and f["pos"] >= 89.47:
            cond.append("A")
        if f["amt"] >= MIN_AMT and f["amt"] <= 37.51 and f["atr"] <= 5.85:
            cond.append("B")
        if "A" not in cond or (code, sig) in seen:   # 모의는 A(밴드 상단)만 추적
            continue
        rec = dict(code=code, name=names.get(code, code), sig=sig,
                   limit=round(f["close"]), cond="+".join(cond),
                   amt=round(f["amt"], 1), pos=round(f["pos"], 1),
                   atr=round(f["atr"], 2))
        picks.append(rec)
        added.append(rec)
        seen.add((code, sig))

    PICKS.write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")

    rows = []
    for p in picks:
        df = load(p["code"])
        if df is None:
            continue
        r = evaluate(p["code"], df, p["sig"], float(p["limit"]))
        if r is None:
            r = dict(status="대기", net=0.0, days=0, entry=None, exit=None)
        rows.append({**p, **r})

    done = [r for r in rows if r["status"] == "청산완료"]
    fills = [r for r in rows if r["status"] != "미체결" and r["status"] != "대기"]
    tot = sum(r["net"] for r in done)
    summary = dict(
        picks=len(rows), filled=len(fills), done=len(done),
        fill_rate=round(len(fills) / max(len(rows), 1) * 100, 1),
        win=round(sum(1 for r in done if r["net"] > 0) / max(len(done), 1) * 100, 1),
        avg=round(tot / max(len(done), 1), 3), tot=round(tot, 1),
        start=min((r["sig"] for r in rows), default=None))

    # 오늘 후보
    todays = [{k: r[k] for k in ("code", "name", "limit", "cond", "amt", "pos", "atr")}
              for r in rows if r["sig"] == today]
    todays.sort(key=lambda t: ("A" not in t["cond"], -t["amt"]))
    for t in todays:
        t["rungs"] = [round(t["limit"] * (1 - GAP) ** k)
                      for k in range(3)]
        t["stop"] = round(t["limit"] * (1 - GAP) ** 2)

    # 청산 유형 분포 + 일별 누적 곡선
    kinds = {}
    for r in done:
        k = r.get("kind") or "기타"
        kinds[k] = kinds.get(k, 0) + 1
    byday = {}
    for r in done:
        d0 = r.get("exit") or r["sig"]
        e = byday.setdefault(d0, dict(n=0, net=0.0))
        e["n"] += 1
        e["net"] += r["net"]
    curve = []
    cum = 0.0
    for d0 in sorted(byday):
        cum += byday[d0]["net"]
        curve.append(dict(day=d0, n=byday[d0]["n"],
                          net=round(byday[d0]["net"], 2), cum=round(cum, 2)))
    summary["kinds"] = kinds
    summary["days"] = len(curve)
    summary["best"] = round(max([r["net"] for r in done], default=0), 2)
    summary["worst"] = round(min([r["net"] for r in done], default=0), 2)

    out = dict(updated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
               curve=curve[-40:],
               sig_date=today, summary=summary,
               today=todays,
               recent=sorted(rows, key=lambda r: -r["sig"])[:12],
               rule=dict(gap=GAP * 100, amts=AMTS, slot=SLOT, hold=HOLD,
                         cond_a="거래대금 ≤ 37.5억 & 밴드위치 ≥ 89.5%",
                         cond_b="거래대금 ≤ 37.5억 & ATR ≤ 5.85%",
                         min_amt=MIN_AMT,
                         entry="전일 종가 지정가", stop="진입일 종가가 3차 가격 아래면 그날 청산"))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[INFO] 신호일 {today} · 오늘 후보 {len(todays)}건 · 신규 {len(added)}건 · "
          f"누적 {len(rows)}건(체결 {len(fills)}) · 완료 {len(done)}건 "
          f"승률 {summary['win']}% 평균 {summary['avg']:+.3f}% 합계 {summary['tot']:+.1f}%")
    for t in todays:
        print(f"   {t['name']:<12} {t['cond']} 지정가 {t['limit']:>8,.0f} "
              f"차수 {t['rungs']} 손절선 {t['stop']:>8,.0f} · 거래대금 {t['amt']}억")


if __name__ == "__main__":
    main()
