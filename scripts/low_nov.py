"""저변동 전략 11월 백테스트 (박스권과 같은 기간·같은 양식) → data/low_nov.json

규칙 (모의매매와 동일):
  - 조건: 20일 평균 거래대금 10~37.5억 & 30일 종가밴드 상단 89.5% 이상 & 우선주 제외
  - 진입: 신호일 종가 지정가, 다음 거래일 저가가 닿으면 체결(시가가 낮으면 시가)
  - 사다리: -1.5% 간격 3차 (6·6·9만원 = 슬롯 21만원), 각 차수 +1.5% 익절
  - 시간손절: 진입일 종가가 3차 가격 아래면 그날 종가 전량 청산
  - 최대 3거래일 보유 후 남은 물량 종가 청산
  - 급락 쿨다운: 전일 KOSPI -3% 이하 → 당일 신규 진입 금지
  - 슬롯 무제한 (박스권 11월 블록과 동일 가정)
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
OUT = ROOT / "data" / "low_nov.json"

SINCE, UNTIL = 20261001, 20261231
GAP, FEE = 0.015, 0.00015
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_AMT, MAX_AMT = 10.0, 37.51
PREF_TAIL = "579K"


def blocked_days():
    m = pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"]).set_index("Date")
    dn = [int(d.strftime("%Y%m%d")) for d in m.index]
    ret = m["Close"].pct_change() * 100
    blk = set()
    for i in range(1, len(dn) - 1):
        if ret.iloc[i] <= -3.0:
            blk.add(dn[i + 1])
    return blk


def trade(df, i):
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    o = df["Open"].values.astype(float)
    if i + 1 >= len(df):
        return None
    lim = float(c[i])
    if float(l[i + 1]) > lim:
        return dict(status="미체결", net=0.0)
    entry_day = int(df.index[i + 1].strftime("%Y%m%d"))
    lv = [lim * (1 - GAP) ** k for k in range(3)]
    tg = [x * (1 + GAP) for x in lv]
    px0 = min(float(o[i + 1]), lim)
    held = {0: px0}
    realised = 0.0
    kind = None
    exit_day = entry_day
    for dd in range(HOLD):
        j = i + 1 + dd
        if j >= len(df):
            break
        for k in (1, 2):
            if k not in held and float(l[j]) <= lv[k]:
                held[k] = lv[k]
        for k in list(held):
            if float(h[j]) >= held[k] * (1 + GAP):
                realised += AMTS[k] * 10_000 * (GAP - FEE)
                del held[k]
        exit_day = int(df.index[min(j, len(df) - 1)].strftime("%Y%m%d"))
        if dd == 0 and held and float(c[j]) < lv[2]:
            for k in list(held):
                realised += AMTS[k] * 10_000 * (float(c[j]) / held[k] - 1 - FEE)
                del held[k]
            kind = "시간손절"
            break
        if not held:
            kind = kind or "전량익절"
            break
    if held:
        j = min(i + HOLD, len(df) - 1)
        for k in list(held):
            realised += AMTS[k] * 10_000 * (float(c[j]) / held[k] - 1 - FEE)
        kind = kind or ("기간청산" if i + HOLD <= len(df) - 1 else "보유중")
        exit_day = int(df.index[j].strftime("%Y%m%d"))
    return dict(status=("보유중" if kind == "보유중" else "청산완료"),
                net=realised, kind=kind or "전량익절",
                entry_day=entry_day, exit_day=exit_day, rungs=len(held) + 0)


def main():
    codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
    master = pd.read_pickle(PIT / "master.pkl")
    names = dict(zip(master["code"].astype(str), master["name"].astype(str)))
    blk = blocked_days()

    out_rows = []
    for code in codes:
        if code[-1] in PREF_TAIL:
            continue
        try:
            df = pd.read_pickle(PIT / f"{code}.pkl")
        except Exception:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df["Volume"] > 0]
        if len(df) < 60:
            continue
        c = df["Close"].values.astype(float)
        v = df["Volume"].values.astype(float)
        cs = pd.Series(c)
        lo30 = cs.rolling(30).min().values
        hi30 = cs.rolling(30).max().values
        amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
        n = len(df)
        _lo = max(40, n - 90)
        for i in range(_lo, n - 1):
            day = int(df.index[i].strftime("%Y%m%d"))
            if not (SINCE <= day <= UNTIL):
                continue
            if i + 1 >= n:
                continue
            ed = int(df.index[i + 1].strftime("%Y%m%d"))
            if ed in blk:                     # 쿨다운 (체결일 기준)
                continue
            if lo30[i] <= 0 or hi30[i] <= lo30[i]:
                continue
            a = amt20[i]
            pos = (c[i] - lo30[i]) / max(hi30[i] - lo30[i], 1e-9) * 100
            if not (MIN_AMT <= a <= MAX_AMT) or pos < 89.47:
                continue
            t = trade(df, i)
            if not t or t["status"] == "미체결":
                continue
            out_rows.append(dict(code=code, name=names.get(code, code), sig=day,
                                 amt=round(a, 1), pos=round(pos, 1), **t))

    rows = out_rows
    done = [r for r in rows if r["status"] == "청산완료"]
    total = sum(r["net"] for r in done)
    entry_cnt = defaultdict(int)
    for r in rows:
        entry_cnt[r["entry_day"]] += 1
    byday = defaultdict(lambda: [0, 0.0])
    for r in done:
        byday[r["exit_day"]][0] += 1
        byday[r["exit_day"]][1] += r["net"]
    daily, cum = [], 0.0
    for d in sorted(byday):
        n, net = byday[d]
        cum += net
        daily.append(dict(d=d, n=n, ein=entry_cnt.get(d, 0), net=round(net),
                          cum=round(cum), pct=round(cum / SLOT * 100, 2)))

    kinds = Counter(r["kind"] for r in rows if r["status"] == "청산완료")
    recent = sorted(rows, key=lambda r: -r["exit_day"])[:12]
    s = dict(positions=len(rows), done=len(done), holding=len(rows) - len(done),
             win=round(sum(1 for r in done if r["net"] > 0) / max(len(done), 1) * 100, 1),
             avg=round(total / max(len(done), 1)), total=round(total),
             pct=round(total / SLOT * 100, 2),
             avg_pct=round(total / max(len(done), 1) / SLOT * 100, 3),
             best=round(max((r["net"] for r in done), default=0)),
             worst=round(min((r["net"] for r in done), default=0)),
             slot=SLOT, kinds=dict(kinds))
    entry_cnt = defaultdict(int)
    for r in rows:
        entry_cnt[r["entry_day"]] += 1
    out = dict(updated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
               entries_by_day=[dict(d=k, n=v) for k, v in sorted(entry_cnt.items())],
               blocked=sorted(x for x in blk if x >= SINCE),
               period=dict(start=min((r["entry_day"] for r in rows), default=None),
                           end=max((r["exit_day"] for r in rows), default=None)),
               summary=s, daily=daily,
               recent=[dict(name=r["name"], entry=r["entry_day"], exit=r["exit_day"],
                            kind=r.get("kind", ""), net=round(r["net"]),
                            pct=round(r["net"] / SLOT * 100, 2),
                            status=r["status"]) for r in recent],
               note=("슬롯 무제한 · 급락 쿨다운 적용(체결일 기준) · 1.5% 3차 · 최대 3일 · "
                     "진입일 종가가 3차 아래면 당일 청산 · 금액은 슬롯 21만원 기준"))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[INFO] 10~11월 진입 {s['positions']}건 · 완료 {s['done']} · 보유 {s['holding']} · "
          f"승률 {s['win']}% · 건당 {s['avg']:,}원({s['avg_pct']:+.3f}%) · "
          f"합계 {s['total']:,}원({s['pct']:+.2f}%)")
    print(f"[INFO] 청산유형 {dict(kinds)} · 곡선 {len(daily)}일")


if __name__ == "__main__":
    main()
