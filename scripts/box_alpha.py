"""박스 전략의 코스피 대비 알파 — 덤프된 포지션 기준

포지션별 net(원)과 진입/청산일을 쓰고, 같은 기간 코스피 수익률을 빼서 알파를 낸다.
포지션 수익률은 net / (슬롯 87만원) 대신, 실제 투입한 금액을 알 수 없으므로
전체 평균 자본(요약값)을 이용해 월 단위로 비교한다.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lucky/box-monitor")
DUMP = Path("/tmp/box_trades.json")
OUT = Path("/tmp/box_out.json")
UNIT = 100_000          # 1차 6만원 등은 설정값, 비교용으로 전체 포지션 금액 사용

trades = json.loads(DUMP.read_text(encoding="utf-8"))
summ = json.loads(OUT.read_text(encoding="utf-8")).get("summary", {})
print(f"[INFO] 포지션 {len(trades):,}건 · 기간 {summ.get('start_year')}~{summ.get('end_year')}")

mkt = pd.read_csv(ROOT / "data" / "ks11_ohlc.csv", parse_dates=["Date"]).set_index("Date")["Close"]
mk = {int(d.strftime("%Y%m%d")): float(v) for d, v in mkt.items()}
mk_days = sorted(mk)
from bisect import bisect_left


def mret(a, b):
    i = bisect_left(mk_days, a)
    if i >= len(mk_days):
        return 0.0
    j = min(bisect_left(mk_days, b), len(mk_days) - 1)
    return (mk[mk_days[j]] / mk[mk_days[i]] - 1) * 100


# 슬롯 금액 추정: 1차 6만 + 2차 6만 + 3차 9만 ... 실제 체결 차수는 알 수 없으므로
# 포지션당 평균 투입액 = 총 순익 / 평균 순익률 대신, 차수 평균을 4차로 가정하지 않고
# '동일 금액 슬롯' 가정: 슬롯 87만원에 net/870000을 수익률로 본다.
SIZES = [6, 6, 9, 12, 18, 18, 18]
SLOT = sum(SIZES) * 10_000
print(f"[INFO] 슬롯 가정 {SLOT:,}원 (1~7차 전액 체결 기준 최대)")

rows = []
for t in trades:
    a, b = t["entry_day"], t["exit_day"]
    rows.append(dict(sig=a, exit=b, net=t["net"],
                     pct=t["net"] / SLOT * 100, days=t["days"],
                     mkt=mret(a, b)))
d = pd.DataFrame(rows)
d["alpha"] = d.pct - d.mkt
print(f"\n포지션당 평균 net   {d.net.mean():+,.0f}원")
print(f"포지션당 수익률     {d.pct.mean():+.3f}% (슬롯 전액 가정)")
print(f"같은 기간 시장      {d.mkt.mean():+.3f}%")
print(f"알파               {d.alpha.mean():+.3f}%p")
print(f"알파 플러스 비율    {(d.alpha>0).mean()*100:.1f}%")

# 실제 체결 차수를 모르므로, 평균 체결 금액을 역산해 보정
# 요약의 평균 투입자본과 포지션 수로 1건당 평균 금액을 구한다
cap = None
try:
    o = json.loads(OUT.read_text(encoding="utf-8"))
    cap = o.get("avg_capital") or o.get("summary", {}).get("avg_capital")
except Exception:
    pass
if cap:
    print(f"[참고] 엔진 평균 투입자본 {cap:,}원")

d["m"] = d.sig // 100
rows = []
for m, g in d.groupby("m"):
    rows.append(dict(month=f"{m//100}-{m%100:02d}", n=len(g), net=g.net.sum(),
                     ret=g.pct.mean(), mkt=g.mkt.mean(), alpha=g.alpha.mean()))
r = pd.DataFrame(rows)
print(f"\n{'월':<9}{'건수':>6}{'건당%':>9}{'시장%':>9}{'알파%p':>9}")
for _, x in r.iterrows():
    print(f"{x.month:<9}{x.n:>6}{x.ret:>+9.3f}{x.mkt:>+9.3f}{x.alpha:>+9.3f}")
print(f"\n알파 플러스 {int((r.alpha>0).sum())}/{len(r)}개월 · 월 평균 알파 {r.alpha.mean():+.3f}%p")
