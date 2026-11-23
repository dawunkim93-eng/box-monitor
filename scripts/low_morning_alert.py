"""아침 매수 안내 — 08:50 텔레그램

전일 15:18에 확정된 후보를 다음 거래일 09:00에 주문한다.
시가 < 전일 종가 → 시장가 / 그 외 → 전일 종가 지정가.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PICKS = ROOT / "data" / "paper_picks.json"
SEAL = ROOT / "data" / "preclose_seal.json"
GAP = 0.015
AMTS = [6, 6, 9]

sys.path.insert(0, str(ROOT / "scripts"))
from low_expiry_alert import load_env, send  # noqa: E402


def main():
    load_env()
    if not PICKS.exists():
        print("[INFO] 원장 없음")
        return
    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    if not picks:
        print("[INFO] 후보 없음")
        return
    seal = {}
    if SEAL.exists():
        try:
            seal = json.loads(SEAL.read_text(encoding="utf-8"))
        except Exception:
            seal = {}
    # 가장 최근 확정일 = 어제 15:18에 만든 후보
    days = sorted({p["sig"] for p in picks}, reverse=True)
    if not days:
        print("[INFO] 후보 없음")
        return
    sig = days[0]
    if str(sig) not in seal:
        print(f"[INFO] {sig}는 장중 확정분이 아님 — 스킵")
        return
    todays = [p for p in picks if p["sig"] == sig]
    if not todays:
        print("[INFO] 대상 없음")
        return

    lines = [f"🌅 {str(sig)[4:6]}/{str(sig)[6:]} 확정 후보 {len(todays)}건",
             "09:00에 아래 가격으로 주문하세요.",
             "· 시가가 전일 종가보다 낮으면 → 시장가 매수",
             "· 높으면 → 전일 종가 지정가 (안 내려오면 미체결)"]
    for p in sorted(todays, key=lambda x: -(x.get("score") or 0)):
        st = round(float(p["limit"]) * (1 - GAP) ** 2)
        auto = round(float(p["limit"]) * 0.965)
        lines.append(f"{p['name']} {p['limit']:,}원 | 3차 {st:,}원 | 자동손절 {auto:,}원")
    lines.append("보유 기간 3거래일 · 각 차수 +2.0% 익절 · 종가가 3차 아래면 청산")
    text = "\n".join(lines)
    if "--no-send" in sys.argv:
        print(text)
        return
    send(text)
    print(f"[INFO] 아침 안내 발송 {len(todays)}건")


if __name__ == "__main__":
    main()
