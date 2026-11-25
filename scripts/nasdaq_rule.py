"""나스닥 급락 다음날 매수 시점 규칙

한국 거래일 D의 아침에 마감한 미국 세션(직전 미국장)이 -2% 이하로 빠졌으면
그날 매수를 장초반(시가)이 아니라 장후반(종가)으로 미룬다.

검증(2년 936건): 장초반 +0.519% → 장후반 +0.706% (건당 +0.187%p, 승률 72.9 → 78.8%)
"""
import csv
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "data" / "nasdaq.csv"
THR = -2.0          # 이 값 이하로 빠지면 종가 매수


@lru_cache(maxsize=1)
def _series():
    rows = []
    if not CSV.exists():
        return [], {}
    with CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                rows.append((r["Date"].replace("-", ""), float(r["Close"])))
            except (KeyError, ValueError):
                continue
    rows.sort()
    days = [d for d, _ in rows]
    ret = {days[i]: (rows[i][1] / rows[i - 1][1] - 1) * 100 for i in range(1, len(rows))}
    return days, ret


def prev_drop(day):
    """한국 거래일(정수 yyyymmdd) 아침에 마감한 미국장 등락률"""
    days, ret = _series()
    if not days:
        return None
    j = bisect_left(days, str(day)) - 1
    if j < 1:
        return None
    return ret.get(days[j])


def shift_to_close(buy_day):
    d = prev_drop(buy_day)
    return (d is not None) and (d <= THR), d


if __name__ == "__main__":
    import datetime
    t = int(datetime.date.today().strftime("%Y%m%d"))
    on, d = shift_to_close(t)
    print(f"{t} 직전 미국장 {d if d is None else round(d, 2)}% → "
          f"{'장후반(종가) 매수' if on else '장초반(시가) 매수'}")
