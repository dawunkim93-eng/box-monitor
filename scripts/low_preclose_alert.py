"""15:20 확정 후보 텔레그램 알림

15:15에 확정한 후보 표를 읽어 텔레그램으로 보낸다.
매수 예정만 보내고, 급등 제외·제외 건수는 요약으로 붙인다.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PICK = ROOT / "data" / "daily_pick.json"

sys.path.insert(0, str(ROOT / "scripts"))
from low_expiry_alert import load_env, send  # noqa: E402


def main():
    load_env()
    if not PICK.exists():
        print("[INFO] daily_pick 없음")
        return
    d = json.loads(PICK.read_text(encoding="utf-8"))
    days = d.get("days") or []
    if not days:
        print("[INFO] 후보 묶음 없음")
        return
    g = days[0]
    sig = g.get("sig")
    items = g.get("items") or []
    buy = [x for x in items if x["status"] in ("매수 대기", "매수 예정")]
    excl = [x for x in items if x["status"] not in ("매수 대기", "매수 예정")]
    if not buy:
        print("[INFO] 매수 예정 없음")
        return

    nd = d.get("nasdaq") or {}
    lines = [f"📋 {str(sig)[4:6]}/{str(sig)[6:]} 확정 후보 · 매수 예정 {len(buy)}건"
             + (f" (제외 {len(excl)}건)" if excl else "")]
    if nd.get("drop") is not None:
        lines.append(f"나스닥 {nd['drop']:+.2f}% → "
                     + ("장후반 15:20 종가 매수" if nd.get("shift") else "장초반 09:00 매수"))
    lines.append("")
    for x in sorted(buy, key=lambda v: -(v.get("score") or 0)):
        lim = x.get("limit") or 0
        r3 = round(lim * 0.985 ** 2)
        st = round(lim * 0.965)
        sc = x.get("score")
        lines.append(f"{x['name']} {lim:,}원 | 3차 {r3:,} | 자동손절 {st:,}"
                     + (f" | {sc:.1f}점" if sc is not None else ""))
    lines.append("")
    lines.append("내일 08:40~09:00 주문 · 시가<매수가면 시장가, 아니면 지정가")
    if excl:
        lines.append("제외: " + ", ".join(f"{x['name']}({x.get('kind') or x['status']})"
                                          for x in excl[:8]))

    text = "\n".join(lines)
    if "--no-send" in sys.argv:
        print(text)
        return
    send(text)
    print(f"[INFO] 15:20 확정 알림 발송 · 매수 예정 {len(buy)}건")


if __name__ == "__main__":
    main()
