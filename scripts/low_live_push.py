"""저변동 표 현재가 실시간 갱신 (장중 15분 간격)

daily_pick.json의 오늘 묶음 + 보유 종목 가격만 갱신한다.
후보·원장·통계는 건드리지 않는다.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "daily_pick.json"
sys.path.insert(0, str(ROOT / "scripts"))
from daily_pick import _live_quotes  # noqa: E402


def main():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    groups = d.get("days") or []
    if not groups:
        print("[INFO] 묶음 없음")
        return
    codes = set()
    for g in groups[:3]:
        for x in g.get("items") or []:
            if x.get("code"):
                codes.add(x["code"])
    codes = sorted(codes)
    if not codes:
        print("[INFO] 대상 없음")
        return
    live = _live_quotes(codes)
    if not live:
        print("[INFO] 시세 수신 실패")
        return
    n = 0
    for g in groups[:3]:
        for x in g.get("items") or []:
            q = live.get(x.get("code"))
            if q:
                x["cur"] = round(q[3])
                if q[0] and not x.get("open"):
                    x["open"] = round(q[0])
                    lim = x.get("limit") or 0
                    if lim:
                        x["gap"] = round((q[0] / lim - 1) * 100, 2)
                n += 1
        # 손절선 이탈 표시 갱신
        for x in g.get("items") or []:
            if x.get("cur") and x.get("stop") and x.get("status") == "보유중":
                x["under"] = bool(x["cur"] < x["stop"])
    d["live_updated"] = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[INFO] 현재가 {n}건 갱신 · {d['live_updated']}")


if __name__ == "__main__":
    main()
