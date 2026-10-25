#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""박스권 Top-20(매수존) 순위 진입 시뮬레이션 — 검증 탭 데이터 (box-monitor 전용)
============================================================================
매일: LIVE 박스(60일 높이 1.25~2.2 · 추세≤20% · ATR≥3% · 20일 연속 · 거래대금
10억+) 중 **매수존(위치≤50%)**만 후보 → priority_score 랭킹 → **상위 20 슬롯** 진입.
포지션: 그리드 사다리 (진입 종가 앵커·ATR티어 갭·러그매도 되팔기·재그리드 허용
· 하단-15% 손절 · 박스 소멸·데이터갭 시 청산) — 슬롯 해제 후 다음 날 최고 순위 진입.

산출(지표): 회전율(연·슬롯당) · 1차 매수가 탈출 확률 · 지속기간(평균/중앙) ·
총·연도별 손익 · 승률 · 최대낙폭 · 청산사유 분포 · 최근 30 포지션 이력.

실행: python scripts/box_validation.py   (매주 토 06:00 크론 · 소요 ~2-4분)
"""
import json
import subprocess
import time as tmod
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "data" / "box_validation.json"
MASTER = Path("/home/lucky/backtest/data/pit/master.pkl")

UNIT, FEE, RUNGS = 100_000, 0.002, 15
K = 20                 # 동시 보유 슬롯 (진입 = 순위 상위 20)
AMT_MIN = 1e9
START = 20160901       # 스캔 시작일


def priority_of(age, pos, atr, drift, ratio):
    """box_alert.priority_score 위임 (없으면 자체 구현 — 스펙 동기 2026-10-23)."""
    try:
        from box_alert import priority_score
        return priority_score(age, pos, atr, drift, ratio)
    except Exception:
        s = 0
        if age is not None:
            s += 4 if age >= 61 else (3 if age >= 31 else 0)
        if pos is not None:
            s += 2 if 0.35 < pos <= 0.5 else (1 if 0.25 < pos <= 0.35 else 0)
        if atr is not None:
            s += 2 if atr >= 0.05 else (1 if atr >= 0.04 else 0)
        if drift is not None and drift > 0:
            s += 1
        if ratio is not None and ratio <= 1.8:
            s += 1
        return s


def load_universe():
    """종목 배열 + 트리거 후보 (매수존·유동성·박스지속)."""
    master = pd.read_pickle(MASTER)
    U = []
    for r in master.itertuples():
        code, name = str(r.code), r.name
        try:
            df = pd.read_pickle(PIT / f"{code}.pkl")
        except Exception:
            continue
        if df is None or len(df) == 0 or not {"Open", "High", "Low", "Close",
                                              "Volume"}.issubset(df.columns):
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df["Volume"] > 0]
        if len(df) < 100:
            continue
        if (df["Close"].pct_change().abs() > 0.60).any():
            continue
        n = len(df)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        c = df["Close"].values.astype(float)
        dn = np.array([int(d.strftime("%Y%m%d")) for d in df.index],
                      dtype=np.int64)
        hi60 = pd.Series(h).rolling(60).max().values
        lo60 = pd.Series(l).rolling(60).min().values
        drift = np.full(n, np.nan)
        drift[60:] = c[60:] / c[:-60] - 1
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)),
                                          np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = (pd.Series(tr).ewm(alpha=1 / 14, min_periods=14).mean().values
               / np.maximum(c, 1))
        amt20 = pd.Series(c * v).rolling(20).mean().values
        ratio = hi60 / np.maximum(lo60, 1e-9)
        box = ((ratio >= 1.25) & (ratio <= 2.2)
               & (np.abs(np.nan_to_num(drift)) <= 0.20)
               & (np.nan_to_num(atr) >= 0.030))
        persist = box & (pd.Series(box.astype(float))
                         .rolling(20).min().values == 1.0)
        pdays = np.zeros(n, dtype=np.int64)
        run = 0
        for i in range(n):
            run = run + 1 if persist[i] else 0
            pdays[i] = run
        posf = np.where(hi60 > lo60, (c - lo60)
                        / np.maximum(hi60 - lo60, 1e-9), 1.0)
        trig = (persist & (posf <= 0.5) & (np.nan_to_num(amt20) >= AMT_MIN)
                & (dn >= START))
        ti = np.where(trig)[0]
        if not len(ti):
            continue
        sc = np.array([priority_of(int(pdays[i]), float(posf[i]),
                                   float(atr[i]) if not np.isnan(atr[i]) else None,
                                   float(drift[i]) if not np.isnan(drift[i]) else 0.0,
                                   float(ratio[i]) if not np.isnan(ratio[i]) else None)
                       for i in ti], dtype=np.int64)
        U.append({"code": code, "name": name, "dn": dn, "h": h, "l": l,
                  "c": c, "atr": atr, "persist": persist, "lo60": lo60,
                  "hi60": hi60, "posf": posf,
                  "trig": ti, "sc": sc})
    return U


def advance_grid(hst, lo, hi, lo60_d, hi60_d, c_d):
    """보유 그리드 하루 진행. 반환: (청산여부, 청산가, 청산사유)."""
    levels, held = hst["levels"], hst["held"]
    stop_px = lo60_d * 0.85
    if lo <= stop_px:
        for k in range(RUNGS):
            if held[k]:
                hst["net"] += (stop_px - levels[k]) / levels[k] * UNIT
                if k == 0 and hst["grid_n"] == 1:
                    hst["loss0"] = True
        return True, stop_px, "손절"
    for k in range(1, RUNGS):
        if not held[k] and lo <= levels[k]:
            held[k] = True
    for k in range(RUNGS):
        if held[k] and hi >= levels[k] * (1 + hst["gap"]):
            hst["net"] += hst["gap"] * UNIT - FEE * UNIT
            held[k] = False
            hst["cycles"] += 1
            if k == 0 and hst["grid_n"] == 1:
                hst["escape0"] = True
    if not any(held):
        pos = ((c_d - lo60_d) / (hi60_d - lo60_d)
               if hi60_d > lo60_d else 1.0)
        if pos <= 0.5:      # 재눌림 → 재그리드
            hst["grid_n"] += 1
            hst["levels"] = [c_d * (1 - hst["gap"]) ** k for k in range(RUNGS)]
            hst["held"] = [False] * RUNGS
            hst["held"][0] = True
    return False, None, None


def main():
    t0 = tmod.time()
    U = load_universe()
    print(f"[INFO] 유니버스 {len(U):,}종목 ({tmod.time()-t0:.0f}s)", flush=True)
    all_days = np.sort(np.unique(np.concatenate([s["dn"] for s in U])))
    # 트리거 이벤트 — (day, -score, stock_i, row_j)
    ev = []
    for i, s in enumerate(U):
        for j, d, sc in zip(s["trig"], (s["dn"][s["trig"]]), s["sc"]):
            ev.append((int(d), -int(sc), i, int(j)))
    ev.sort()
    ev_days = defaultdict(list)
    for e in ev:
        ev_days[e[0]].append(e)

    holdings = {}            # stock_i → hst
    done, curve = [], []     # curve: (exit_day, realized)
    for D in all_days:
        if D < START:
            continue
        for i in list(holdings):
            hst, s = holdings[i], U[i]
            dn = s["dn"]
            j = int(np.searchsorted(dn, D))
            if j >= len(dn) or dn[j] != D:
                close_position(done, holdings, curve, i, float(s["c"][hst["j"]]),
                               "데이터갭", D)
                continue
            if not bool(s["persist"][j]):
                close_position(done, holdings, curve, i, float(s["c"][j]),
                               "박스소멸", D)
                continue
            hst["j"] = j
            stopQ, px, kind = advance_grid(hst, float(s["l"][j]),
                                           float(s["h"][j]),
                                           float(s["lo60"][j]),
                                           float(s["hi60"][j]),
                                           float(s["c"][j]))
            if stopQ:
                close_position(done, holdings, curve, i, px, kind, D)
        free = K - len(holdings)
        if free > 0:
            for d, nsc, i, j in ev_days.get(int(D), []):
                if free <= 0:
                    break
                if i in holdings:
                    continue
                s = U[i]
                atr = float(s["atr"][j])
                g = tier_gap(atr) if atr is not None and not np.isnan(atr) else 0.02
                holdings[i] = {"j": j, "entry_day": int(D),
                               "entry": float(s["c"][j]),
                               "score": -nsc,
                               "gap": g,
                               "levels": [float(s["c"][j]) * (1 - g) ** k
                                          for k in range(RUNGS)],
                               "held": [False] * RUNGS,
                               "j_entry": j, "grid_n": 1, "escape0": False, "loss0": False,
                               "cycles": 0, "net": 0.0}
                free -= 1
    # 만기 데이터 종료 — 잔여 포지션 청산
    for i in list(holdings):
        s = U[i]
        close_position(done, holdings, curve, i, float(s["c"][holdings[i]["j"]]),
                       "데이터종료", int(all_days[-1]))
    print(f"[INFO] 워크 완료 · 포지션 {len(done):,} ({tmod.time()-t0:.0f}s)",
          flush=True)
    finish(done, curve, all_days, U)


def close_position(done, holdings, curve, i, px, kind, day):
    hst = holdings.pop(i)
    done.append({"code": i, "entry_day": hst["entry_day"], "exit_day": int(day),
                 "entry": hst["entry"], "score": hst["score"],
                 "escape0": hst["escape0"], "loss0": hst["loss0"],
                 "cycles": hst["cycles"], "net": hst["net"],
                 "days": hst["j"] - hst["j_entry"],
                 "kind": kind})
    curve.append((int(day), hst["net"]))


def finish(done, curve, all_days, U):
    """지표 집계 + 저장 + push."""
    y0 = int(all_days[0]) // 10000
    y1 = int(all_days[-1]) // 10000
    years = max((y1 - y0) + 1, 1)
    # j_entry 보정 (close_position에서 사용)
    for p in done:
        if "days" not in p:
            p["days"] = 0
    nets = [p["net"] for p in done]
    esc = sum(1 for p in done if p["escape0"]) / max(len(done), 1) * 100
    win = sum(1 for x in nets if x > 0) / max(len(nets), 1) * 100
    eq, peak, dd = 0.0, 0.0, 0.0
    for d, v in sorted(curve, key=lambda x: x[0]):
        eq += v
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    yy = defaultdict(lambda: {"net": 0.0, "n": 0, "esc": 0})
    for p in done:
        yv = int(p["exit_day"]) // 10000
        yy[yv]["net"] += p["net"]
        yy[yv]["n"] += 1
        if p["escape0"]:
            yy[yv]["esc"] += 1
    kinds = Counter(p["kind"] for p in done)
    name_of = {i: s["name"] for i, s in enumerate(U)}
    recent = sorted(done, key=lambda p: p["exit_day"],
                    reverse=True)[:30]
    out = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": {"slots": K, "buy_zone": "위치≤50%", "unit": "10만/차",
                   "gap": "ATR티어 1~4%", "start": str(y0), "end": str(y1)},
        "summary": {
            "positions": len(done),
            "years": years,
            "turnover_year": round(len(done) / years, 1),
            "turnover_slot": round(len(done) / years / K, 2),
            "escape_rate": round(esc, 1),
            "loss0_rate": round(sum(1 for p in done if p["loss0"]) / max(len(done), 1) * 100, 1),
            "win_rate": round(win, 1),
            "hold_avg": round(sum(p["days"] for p in done)
                              / max(len(done), 1), 1),
            "hold_median": int(sorted(p["days"] for p in done)
                               [len(done) // 2]) if done else 0,
            "cycles_avg": round(sum(p["cycles"] for p in done)
                                / max(len(done), 1), 1),
            "net_total": round(sum(nets)),
            "mdd_won": round(dd),
            "loss_years": sum(1 for v in yy.values() if v["net"] < 0),
            "n_years": len(yy),
        },
        "yearly": [{"year": y, "net": round(v["net"]),
                    "positions": v["n"],
                    "esc_rate": round(v["esc"] / v["n"] * 100, 1) if v["n"] else None}
                   for y, v in sorted(yy.items())],
        "exit_kinds": dict(kinds),
        "recent_positions": [
            {"name": name_of.get(p["code"], "?"), "entry_day": str(p["entry_day"]),
             "exit_day": str(p["exit_day"]), "days": p["days"],
             "score": p["score"], "escaped": p["escape0"],
             "cycles": p["cycles"], "net": round(p["net"]), "kind": p["kind"]}
            for p in recent],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    subprocess.run(["git", "-C", str(HERE.parent), "add", "-A"],
                   capture_output=True)
    subprocess.run(["git", "-C", str(HERE.parent), "commit", "-m",
                    f"chore: 시뮬 갱신 (포지션 {len(done)})"], capture_output=True)
    subprocess.run(["git", "-C", str(HERE.parent), "push"], capture_output=True)
    s = out["summary"]
    print(f"[INFO] 포지션 {s['positions']} · 탈출율 {s['escape_rate']}% · "
          f"건당회전/년 {s['turnover_year']} · 지속 {s['hold_avg']}일 · "
          f"총 {s['net_total']:+,}원 · 적자연도 {s['loss_years']}/{s['n_years']} "
          f"({tmod.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()