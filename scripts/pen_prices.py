"""연금 보유분 가격 피드 — 네이버 ETF 목록(1,184종목)을 받아 data/pen_prices.json 생성."""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

OUT = "/home/lucky/box-monitor/data/pen_prices.json"
URL = "https://finance.naver.com/api/sise/etfItemList.nhn"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Referer": "https://finance.naver.com/sise/etf.naver"}
KST = timezone(timedelta(hours=9))


def main():
    r = requests.get(URL, headers=UA, timeout=20)
    raw = r.content
    d = None
    for enc in ("euc-kr", "cp949", "utf-8"):
        try:
            d = json.loads(raw.decode(enc))
            break
        except Exception:
            continue
    if not d or "result" not in d:
        print("fetch failed", file=sys.stderr)
        return 1
    items = []
    for x in d["result"]["etfItemList"]:
        code = x.get("itemcode")
        price = x.get("nowVal")
        if not code or not price:
            continue
        items.append({
            "c": code,
            "n": x.get("itemname", ""),
            "p": price,
            "r": x.get("changeRate"),
            "nav": x.get("nav"),
            "ms": x.get("marketSum"),
            "q3": x.get("threeMonthEarnRate"),
        })
    now = datetime.now(KST)
    out = {
        "asof": now.strftime("%Y-%m-%d %H:%M"),
        "date": now.strftime("%Y-%m-%d"),
        "source": "naver etf list (지연 시세)",
        "count": len(items),
        "items": items,
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, OUT)
    print(f"{now:%H:%M} {len(items)}종목 → {OUT} ({os.path.getsize(OUT)/1024:.0f}KB)")


if __name__ == "__main__":
    sys.exit(main())
