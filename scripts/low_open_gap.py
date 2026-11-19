"""저변동 후보의 '진입일' 시가·갭 수집 (평일 09:05)

후보는 전일 종가로 만들어지고, 실제 진입일은 그 다음 거래일이다.
사이트에는 진입일 시가와 (전일 종가 대비) 갭을 보여줘야 하므로
장 시작 직후 오늘 시가를 받아 저장한다.
산출: data/low_open.json
"""
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
PICKS = HERE / "data" / "paper_picks.json"
OUT = HERE / "data" / "low_open.json"
UA = {"User-Agent": "Mozilla/5.0"}


def today_open(code, day):
    try:
        st = (pd.Timestamp.now() - pd.Timedelta(days=6)).strftime("%Y%m%d")
        url = ("https://api.finance.naver.com/siseJson.naver?"
               f"symbol={code}&requestType=1&startTime={st}&endTime={day}&timeframe=day")
        raw = urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=8).read().decode("utf-8", "ignore")
        rows = re.findall(r'\["(\d{8})",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', raw)
        for d, o, h, l, c in rows:
            if int(d) == int(day):
                return float(o), float(c)
        return None, None
    except Exception:
        return None, None


def main():
    if not PICKS.exists():
        print("[INFO] 원장 없음")
        return
    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    day = int(datetime.now().strftime("%Y%m%d"))

    # 오늘 진입 대상 = 신호일이 오늘 이전인 마지막 후보들
    last_sig = max((p["sig"] for p in picks), default=None)
    target = [p for p in picks if p["sig"] == last_sig] if last_sig else []
    items = []
    for p in target:
        o, c = today_open(p["code"], day)
        if o is None:
            continue
        items.append(dict(code=p["code"], name=p["name"], sig=p["sig"],
                          limit=p["limit"], open=round(o),
                          gap=round((o / max(float(p["limit"]), 1e-9) - 1) * 100, 2)))
    out = dict(updated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
               date=day, sig=last_sig, count=len(items), items=items)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[INFO] {day} 진입 대상 {len(items)}건 · 시가·갭 저장")
    for x in items[:6]:
        print(f"   {x['name']:<12} 지정가 {x['limit']:>8,} 시가 {x['open']:>8,} "
              f"갭 {x['gap']:+.2f}%")


if __name__ == "__main__":
    main()
