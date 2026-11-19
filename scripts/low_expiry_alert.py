"""저변동 만기·손절선 청산 임박 알림 (평일 15:15)

대상:
  · 만기 청산 — 오늘이 3번째 보유일이고 아직 안 팔린 차수가 남은 건
  · 손절선 아래 — 현재가가 3차(-3%) 아래라 종가 청산될 가능성이 큰 건
발송: 텔레그램 (--no-send 로 미리보기)
"""
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
PICKS = HERE / "data" / "paper_picks.json"
STATE = HERE / "data" / "low_alert_state.json"
PIT = Path("/home/lucky/backtest/data/pit")
ENV = Path("/home/lucky/magicsplit-safe-strategy/backtest/.env")

UA = {"User-Agent": "Mozilla/5.0"}
GAP = 0.015
AMTS = [6, 6, 9]
HOLD = 3


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
        "parse_mode": "HTML", "disable_web_page_preview": "true"}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage",
                                 data=payload)
    with urllib.request.urlopen(req, timeout=30) as r:
        return bool(json.loads(r.read().decode()).get("ok"))


def now_price(code):
    try:
        req = urllib.request.Request(
            f"https://m.stock.naver.com/api/stock/{code}/basic",
            headers={"User-Agent": "Mozilla/5.0"})
        d = json.loads(urllib.request.urlopen(req, timeout=6).read().decode())
        return float(str(d.get("closePrice", "")).replace(",", "")) or None
    except Exception:
        return None


def today_bar(code):
    """네이버에서 오늘 봉 (날짜, 시가, 고가, 저가, 종가) — 없으면 None"""
    try:
        end = pd.Timestamp.now()
        st = (end - pd.Timedelta("10D")).strftime("%Y%m%d")
        url = ("https://api.finance.naver.com/siseJson.naver?"
               f"symbol={code}&requestType=1&startTime={st}&endTime={end:%Y%m%d}&timeframe=day")
        raw = urllib.request.urlopen(urllib.request.Request(
            url, headers=UA), timeout=8).read().decode("utf-8", "ignore")
        rows = re.findall(r'\["(\d{8})",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', raw)
        if not rows:
            return None
        d, o, h, l, c = rows[-1]
        return int(d), float(o), float(h), float(l), float(c)
    except Exception:
        return None


def load(code):
    try:
        df = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Volume"] > 0]
    return df if len(df) > 30 else None


def prev_close(code):
    """전일 종가 (당일 등락률 표시용)"""
    try:
        end = pd.Timestamp.now()
        st = (end - pd.Timedelta("10D")).strftime("%Y%m%d")
        url = ("https://api.finance.naver.com/siseJson.naver?"
               f"symbol={code}&requestType=1&startTime={st}&endTime={end:%Y%m%d}&timeframe=day")
        raw = urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}), timeout=8).read().decode("utf-8", "ignore")
        rows = re.findall(r'\["(\d{8})",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', raw)
        return float(rows[-2][4]) if len(rows) >= 2 else None
    except Exception:
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-send", action="store_true")
    ap.add_argument("--force", action="store_true", help="오늘 이미 보냈어도 다시 보냄")
    a = ap.parse_args()
    load_env()

    if not PICKS.exists():
        print("[INFO] 원장 없음")
        return
    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    today = datetime.now().strftime("%Y%m%d")

    rows = []
    for p in picks:
        df = load(p["code"])
        if df is None:
            continue
        tb = today_bar(p["code"])
        if tb and tb[0] > int(df.index[-1].strftime("%Y%m%d")):
            row = pd.DataFrame([{"Open": tb[1], "High": tb[2], "Low": tb[3],
                                 "Close": tb[4], "Volume": 1.0}],
                               index=pd.DatetimeIndex([pd.Timestamp(str(tb[0]))]))
            df = pd.concat([df, row])
        days = [int(d.strftime("%Y%m%d")) for d in df.index]
        if p["sig"] not in days:
            continue
        i = days.index(p["sig"])
        e = i + 1
        if e >= len(df):          # 아직 체결 전
            continue
        lim = float(p["limit"])
        if float(df["Low"].values[e]) > lim:      # 미체결
            continue
        hold_days = (len(df) - e) + 1             # 오늘이 몇 일째 (1=진입일)
        if hold_days > HOLD:
            continue
        lv = [lim * (1 - GAP) ** k for k in range(3)]
        held = {0: min(float(df["Open"].values[e]), lim)}
        for k in (1, 2):
            if float(df["Low"].values[e]) <= lv[k]:
                held[k] = lv[k]
        # 진입일 이후 매도 반영
        for j in range(e, len(df)):
            for k in list(held):
                if float(df["High"].values[j]) >= held[k] * (1 + GAP):
                    del held[k]
        if not held:
            continue                              # 이미 전량 익절
        cur = now_price(p["code"])
        if cur is None:
            continue
        for k in (1, 2):                     # 오늘 현재가로 추가 체결 추정
            if k not in held and cur <= lv[k]:
                held[k] = lv[k]
        if not held:
            continue
        pc = prev_close(p["code"])
        base = sum(AMTS[k] * 10_000 for k in held)
        pnl = sum(AMTS[k] * 10_000 * (cur / held[k] - 1) for k in held)
        row = dict(name=p["name"], code=p["code"], limit=lim, cur=cur,
                   stop=round(lv[2]), left=len(held), pnl=round(pnl),
                   pct=round(pnl / base * 100, 2) if base else 0,
                   hold=hold_days, sig=p["sig"],
                   chg=round((cur / pc - 1) * 100, 2) if pc else None,
                   depth=round((cur / lv[2] - 1) * 100, 2),
                   under=cur < lv[2], exp=(hold_days >= HOLD))
        if row["under"] or row["exp"]:
            rows.append(row)

    if not rows:
        today_s = today
        print(f"[INFO] {today_s} 청산 임박 없음")
        return

    rows.sort(key=lambda r: (r["depth"], -r["pnl"]))
    cut = [r for r in rows if r["under"] and not r["exp"]]
    both = [r for r in rows if r["under"] and r["exp"]]
    exp = [r for r in rows if r["exp"] and not r["under"]]

    def line(r):
        chg = f"{r['chg']:+.1f}%" if r["chg"] is not None else "-"
        return (f"· <b>{r['name']}</b> 현재 {r['cur']:,.0f} ({chg}) · "
                f"3차 {r['stop']:,} 대비 {r['depth']:+.1f}% · 남은 {r['left']}차 · "
                f"{r['pct']:+.1f}%")

    lines = [f"⏰ <b>저변동 청산 임박</b> ({datetime.now():%m/%d %H:%M})"]
    if cut:
        lines.append(f"\n⛔ <b>손절선 아래</b> — 종가도 3차 밑이면 오늘 청산 ({len(cut)}건)")
        lines += [line(r) for r in cut]
    if both:
        lines.append(f"\n⛔⏰ <b>손절선 아래 + 만기</b> — 오늘 반드시 정리 ({len(both)}건)")
        lines += [line(r) for r in both]
    if exp:
        lines.append(f"\n⏰ <b>만기 청산</b> — {HOLD}일째, 오늘 종가에 정리 ({len(exp)}건)")
        lines += [line(r) for r in exp]
    lines.append("\n장 마감 20분 전까지 종가를 확인하고 정리하세요.")
    msg = "\n".join(lines)
    print(msg)

    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    if st.get("last") == today and not a.force:
        print("[INFO] 오늘 이미 알림 발송 — 생략")
        return
    if a.no_send:
        print("[INFO] 전송 생략 (--no-send)")
        return
    if send(msg):
        STATE.write_text(json.dumps({"last": today}), encoding="utf-8")
        print("[INFO] 텔레그램 전송 완료")


if __name__ == "__main__":
    main()
