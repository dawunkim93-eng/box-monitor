"""데일리 복기용 스냅샷 — 그날의 신호(후보, 진입, 청산, 경고)를 모아
data/box_review.json 으로 저장한다. 장 마감 확정(16:10) 이후 실행.
"""
import json
import os
import sys
from datetime import datetime

LIVE = "/home/lucky/box-monitor/data/box_live.json"
LEDGER = "/home/lucky/magicsplit-safe-strategy/backtest/results/box_ledger.jsonl"
OUT = "/home/lucky/box-monitor/data/box_review.json"


def main():
    live = json.load(open(LIVE, encoding="utf-8"))
    rows = live.get("rows") or []
    date = (live.get("updated_at_kst") or "")[:10]

    cands = [{
        "code": r["code"], "name": r["name"], "price": r.get("price"),
        "score": r.get("score"), "pos": round(r.get("pos") or 0, 1),
        "pct_low": round(r.get("pct_low") or 0, 1), "bline": r.get("bline"),
        "gap": r.get("gap"),
    } for r in rows if r.get("cand")]

    enters, exits = [], []
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if not (e.get("date") or "").startswith(date):
                continue
            if e.get("event") == "enter":
                enters.append({"code": e.get("ticker"), "name": e.get("name"),
                               "price": e.get("price"), "score": e.get("score"),
                               "pos": round((e.get("pos") or 0) * 100, 1)})
            elif e.get("event") == "exit":
                exits.append({"code": e.get("ticker"), "name": e.get("name"),
                              "reason": e.get("reason")})

    alerts = [{"code": a["code"], "name": a["name"], "price": a.get("price"),
               "lo": a.get("lo"), "pct_low": round(a.get("pct_low") or 0, 2),
               "time": a.get("time"), "deep": a.get("deep")}
              for a in (live.get("alerts") or [])]

    payload = {
        "date": date,
        "asof": live.get("updated_at_kst"),
        "market": live.get("market"),
        "cooldown": live.get("cooldown") or {"on": False},
        "live_count": live.get("live_count"),
        "candidates": cands,
        "enters": enters,
        "exits": exits,
        "alerts": alerts,
        "holdings": list((live.get("holdings") or {}).keys()),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, OUT)
    print(f"{date} 후보 {len(cands)} · 진입 {len(enters)} · 청산 {len(exits)} · 경고 {len(alerts)} → {OUT}")


if __name__ == "__main__":
    sys.exit(main())
