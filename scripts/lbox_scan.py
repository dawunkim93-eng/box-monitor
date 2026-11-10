#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""지지선 박스(L-box) 관찰 목록 — data/box_lbox.json
=======================================================================
1년(52주) 주봉 창에서
  · 지지선 = 창 안 최저 종가
  · 저항선 = 창 안 최고가
  · 조건: 폭 20~60% · 지지선 ±3% 터치 후 위로 마감 3회 이상
         · 상단을 종가로 돌파한 적 없음 · 최근 26주 지지선 아래 마감 없음
  · 현재가가 지지선 위 8% 이내
매매 신호가 아니라 관찰 목록이다. 진입 규칙은 기존 사다리(주봉 박스) 쪽을 따른다.
실행: python scripts/lbox_scan.py
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PIT = Path("/home/lucky/backtest/data/pit")
MASTER = PIT / "master.pkl"
OUT = Path(__file__).resolve().parent.parent / "data" / "box_lbox.json"

WIN = 52
TOUCH = 0.03
MIN_TOUCH = 3
ENTRY_MAX = 0.08
W_LO, W_HI = 0.20, 0.60
HOLD_RECENT = 26
S_STABLE = 0.97      # 13주 전 지지선 대비 3% 넘게 낮아지지 않았을 것
RECENT_TOUCH = 13    # 최근 13주 안에 지지선 터치 1회 이상


def check(df):
    w = pd.DataFrame({
        "o": df["Open"].resample("W-FRI").first(),
        "h": df["High"].resample("W-FRI").max(),
        "l": df["Low"].resample("W-FRI").min(),
        "c": df["Close"].resample("W-FRI").last(),
    }).dropna()
    if len(w) < WIN:
        return None
    win = w.iloc[-WIN:]
    # 지지선은 최근 4주를 뺀 나머지 48주에서 형성된 바닥 (신저가 종목 제외)
    s = float(win["c"].iloc[:-4].min())
    r = float(win["h"].max())
    px = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else px
    if s <= 0 or r <= s:
        return None
    width = r / s - 1
    if not (W_LO <= width <= W_HI):
        return None
    if not (s <= px <= s * (1 + ENTRY_MAX)):
        return None
    touch = int(np.sum((win["l"] <= s * (1 + TOUCH)) & (win["l"] >= s * (1 - TOUCH))
                        & (win["c"] >= s * 1.005)))
    if touch < MIN_TOUCH:
        return None
    if np.any(win["c"] > r * 1.01):
        return None
    if len(win) >= HOLD_RECENT and float(win["c"].iloc[-HOLD_RECENT:].min()) < s * 0.99:
        return None
    # 지지선이 무너지고 있지 않은지 (13주 전 창의 지지선과 비교)
    if len(w) >= WIN + 13:
        s_prev = float(w["c"].iloc[-WIN - 13:-13].min())
        if s_prev > 0 and s < s_prev * S_STABLE:
            return None
    # 최근 13주 안에 실제로 지지선을 지킨 적이 있는지
    rec = win.iloc[-RECENT_TOUCH:]
    touch_recent = int(np.sum((rec["l"] <= s * (1 + TOUCH)) & (rec["l"] >= s * (1 - TOUCH))
                               & (rec["c"] >= s * 1.005)))
    if touch_recent < 1:
        return None
    return dict(px=round(px), s=round(s), r=round(r),
                width=round(width * 100, 1), pos=round((px - s) / (r - s) * 100, 1),
                room=round((r / px - 1) * 100, 1), touch=touch,
                rtouch=touch_recent,
                chg=round((px / prev - 1) * 100, 2) if prev > 0 else 0.0)


def main():
    master = pd.read_pickle(MASTER)
    rows = []
    for rec in master.itertuples():
        code, name = str(rec.code), str(rec.name)
        try:
            df = pd.read_pickle(PIT / f"{code}.pkl")
        except Exception:
            continue
        if df is None or len(df) < 300:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df["Volume"] > 0]
        if len(df) < 300 or not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
            continue
        try:
            r = check(df)
        except Exception:
            continue
        if r:
            r.update(code=code, name=name)
            rows.append(r)
    rows.sort(key=lambda x: (-x["room"], -x["touch"]))
    out = {
        "updated_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M"),
        "params": {"win": WIN, "touch": TOUCH * 100, "min_touch": MIN_TOUCH,
                   "entry_max": ENTRY_MAX * 100, "w_lo": W_LO * 100, "w_hi": W_HI * 100},
        "count": len(rows),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[OK] 지지선 박스 {len(rows)}종목 → {OUT}")


if __name__ == "__main__":
    main()
