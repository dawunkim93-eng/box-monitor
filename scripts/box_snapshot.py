#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""박스권 순위 스냅샷 — 매일 아침 상위 20 기록 (검증 탭용, box-monitor 전용)
============================================================================
아침 box_alert(08:15)가 저장한 상태(results/box_state.json)에서 점수·순위를
재계산해 data/box_rank_history.json (90일 롤링)에 일 1행 추가 후 push.

실행: python scripts/box_snapshot.py   (평일 08:25 크론 — box_alert 직후)
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent          # box-monitor/scripts
REPO = HERE.parent                              # box-monitor
SAFE_BT = Path("/home/lucky/magicsplit-safe-strategy/backtest")
sys.path.insert(0, str(SAFE_BT))
from box_alert import priority_score, esc_p  # noqa: E402

STATE = SAFE_BT / "results" / "box_state.json"
HIST = REPO / "data" / "box_rank_history.json"
MASTER = Path("/home/lucky/backtest/data/pit/master.pkl")
TOP = 20
KEEP = 90


def load_names():
    import pandas as pd
    return {str(r.code): r.name for r in pd.read_pickle(MASTER).itertuples()}


def git_push(msg):
    try:
        subprocess.run(["git", "-C", str(REPO), "add", "-A"],
                       capture_output=True)
        subprocess.run(["git", "-C", str(REPO), "commit", "-m", msg],
                       capture_output=True)
        for _ in range(3):
            r = subprocess.run(["git", "-C", str(REPO), "push"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return True
            subprocess.run(["git", "-C", str(REPO), "pull", "--rebase",
                            "--autostash"], capture_output=True)
    except Exception as e:
        print(f"[WARN] git 실패: {e}", file=sys.stderr)
    return False


def main():
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    st = json.loads(STATE.read_text(encoding="utf-8"))
    if "--force" not in sys.argv:
        if not str(st.get("saved_at", "")).startswith(today):
            print(f"[WARN] 상태가 오늘 아침 아님 ({st.get('saved_at')}) — 스킵")
            return
    else:
        today = str(st.get("saved_at", today))[:10]   # 상태 자체 날짜 기록
    live = st.get("live", {})
    if not live:
        print("[INFO] LIVE 0종목 — 스킵")
        return
    names = load_names()
    rows = []
    for c, rec in live.items():
        lo, hi, px = rec.get("lo"), rec.get("hi"), rec.get("px")
        pos = ((px - lo) / (hi - lo)
               if (lo is not None and hi and hi > lo) else None)
        sc = priority_score(rec.get("days"), pos, rec.get("atr"),
                            rec.get("drift"), rec.get("ratio"))
        rows.append({"code": c, "name": names.get(c, c), "score": sc,
                     "esc_p": esc_p(sc),
                     "price": round(px, 2) if px is not None else None,
                     "pos": round(pos * 100, 1) if pos is not None else None,
                     "atr": (round(rec["atr"] * 100, 1)
                             if rec.get("atr") else None),
                     "days": rec.get("days"),
                     "strong": bool(rec.get("strong")),
                     "risk": bool(rec.get("risk"))})
    rows.sort(key=lambda r: (-r["score"], -(r.get("atr") or 0)))
    top = rows[:TOP]

    hist = []
    if HIST.exists():
        try:
            hist = json.loads(HIST.read_text(encoding="utf-8"))
        except Exception:
            hist = []
    if hist and hist[-1].get("date") == today:
        hist[-1]["rows"] = top          # 재실행 → 당일 행 교체 (멱등)
    else:
        hist.append({"date": today, "rows": top})
    hist = hist[-KEEP:]
    prev_rows = ({r["code"]: r for r in hist[-2]["rows"]}
                 if len(hist) > 1 else {})
    cur = hist[-1]["rows"]
    for i, r in enumerate(cur, 1):
        p = prev_rows.get(r["code"])
        r["rank"] = i
        r["prev_rank"] = p["rank"] if p else None
        r["chg_pct"] = (round((r["price"] / p["price"] - 1) * 100, 1)
                        if (p and p.get("price")) else None)
    HIST.parent.mkdir(parents=True, exist_ok=True)
    HIST.write_text(json.dumps(hist, ensure_ascii=False), encoding="utf-8")
    ok = git_push(f"chore: 박스 순위 스냅샷 {today} (상위 {len(top)})")
    print(f"[INFO] 스냅샷 {len(top)}종목 기록 (누적 {len(hist)}일) · "
          f"push {'✅' if ok else '❌'}")


if __name__ == "__main__":
    main()