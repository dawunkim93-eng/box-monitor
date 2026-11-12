#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""진입 가능 종목이 새로 생기면 텔레그램으로 알린다.
=====================================================================
· 사이트 데이터(data/box_live.json)의 cand 플래그를 15분마다 확인
· 그날 처음 뜬 종목만 전송 (종목당 하루 1회)
· 급락 쿨다운이면 전송하지 않음 (신규 진입 금지일)
실행: python3 scripts/box_cand_alert.py [--dry-run] [--test]
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent.parent
LIVE = HERE / "data" / "box_live.json"
STATE = HERE / "data" / "box_cand_state.json"
ENV = Path("/home/lucky/magicsplit-safe-strategy/backtest/.env")


def load_env():
    if not ENV.exists():
        return
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def send(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[ERROR] 텔레그램 인증 없음")
        return False
    payload = urllib.parse.urlencode({
        "chat_id": chat, "text": text,
        "parse_mode": "HTML", "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=payload)
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read().decode())
    if not body.get("ok"):
        print(f"[ERROR] 전송 실패: {body}")
        return False
    return True


def fmt(n):
    return f"{n:,.0f}" if isinstance(n, (int, float)) else "-"


def main():
    args = set(sys.argv[1:])
    dry = "--dry-run" in args
    test = "--test" in args
    if not LIVE.exists():
        print("[WARN] box_live.json 없음")
        return
    d = json.loads(LIVE.read_text(encoding="utf-8"))
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    today = now.strftime("%Y-%m-%d")
    rows = [r for r in d.get("rows", []) if r.get("cand")]
    cd = d.get("cooldown") or {}
    if cd.get("on") and not test:
        print(f"[INFO] 급락 쿨다운 중(전일 {cd.get('pct')}%) — 신규 진입 알림 생략")
        return
    if not rows and "--sample" not in args:
        print("[INFO] 진입 가능 0종목 — 무음")
        return
    if "--sample" in args:
        rows = [{"code": "000000", "name": "예시종목", "price": 12500, "limit": 12300,
                 "pct_low": 7.4, "height": 0.241, "score": 4}]
    st = {}
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            st = {}
    if st.get("date") != today:
        st = {"date": today, "active": []}
    # 직전 확인 때 '진입 가능'이던 종목 — 빠졌다가 다시 뜨면 새 알림으로 본다
    active = set(st.get("active", st.get("sent", [])))
    new = rows if test else [r for r in rows if r["code"] not in active]
    if not new:
        print(f"[INFO] 새 진입 가능 0건 (직전 확인 {len(active)}종목 유지)")
        # 현재 목록으로 상태 갱신 (빠진 종목 추적)
        if not test:
            STATE.write_text(json.dumps(
                {"date": today, "active": sorted({r["code"] for r in rows})},
                ensure_ascii=False), encoding="utf-8")
        return
    head = ("🧪 <b>[테스트] 알림 형식 예시</b>\n\n" if "--sample" in args else "")
    lines = [f"🎯 <b>진입 가능 {len(new)}종목</b> {now:%m-%d %H:%M}",
             "다음 거래일 지정가 주문 · 시가가 낮으면 시가 체결, 높으면 스킵", ""]
    for r in new[:20]:
        lines.append(
            f"· <b>{r['name']}</b> {fmt(r.get('price'))}원 · 지정가 <b>{fmt(r.get('limit'))}</b>"
            f"\n   하단대비 {r.get('pct_low')}% · 폭 {(r.get('height') or 0)*100:.0f}%"
            f" · 점수 {r.get('score')}"
            + (f"\n   🟢 매물대 위 ×2 · 사다리 {('·'.join(str(x) for x in r['amts']))}만원"
               if r.get("boost") == 2 and r.get("amts")
               else f"\n   사다리 {('·'.join(str(x) for x in r.get('amts') or [6,6,9,12,18]))}만원"))
    if len(new) > 20:
        lines.append(f"… 외 {len(new)-20}종목")
    lines += ["", "※ 관찰 알림이며 매수 권장이 아닙니다."]
    msg = head + "\n".join(lines)
    if dry:
        print(msg)
        return
    load_env()
    if send(msg):
        print(f"[INFO] 텔레그램 전송 · {len(new)}종목")
        if not test and "--sample" not in args:
            STATE.write_text(json.dumps(
                {"date": today, "active": sorted({r["code"] for r in rows})},
                ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
