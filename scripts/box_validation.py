#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""박스권 Top-20(매수존) 순위 진입 시뮬레이션 — 검증 탭 데이터 (box-monitor 전용)
============================================================================
확정 구성 (2026-10-30 · C안 — 12년 재검증 산출):
  진입   : LIVE 박스(60일 높이 1.25~2.2 · 추세≤20% · ATR≥3% · 20일 연속 · 거래대금
           10억+) 중 매수존(위치≤50%) → priority_score 순위 상위 20 슬롯
           · 200일선(40주 완성 주봉) 하회 제외 · 이탈일(종가<전일 하단) 진입 금지
           · 신호일 종가 알림 → 다음 영업일 시가 매수 (실전 타이밍)
  사다리 : 진입가 앵커 · 갭 7% 고정 · 5차 (0.375 커버) · 러그매도 되팔기 · 재그리드
           · 당일 신규체결 차수는 종가≥목표가일 때만 매도 (지정가 사다리 현실 가정)
  청산   : ① 종가확정 즉시(종가<전일 박스 하단 → 다음날 시가) ② 시간중단 15 영업일
           ③ 박스 소멸 ④ -15% 백스톱(전일 하단 기준) — 모두 다음 영업일 시가 체결

■ 산출: 회전 · 탈출율(첫 차수 +갭 매도 비율) · 지속 · 연도별 손익 · 승률 · MDD
   · 청산사유(방향 분리) · 점수구간별 탈출율(score_escape) · 투입/예약자본 수익률
   · 가정 명시(assumptions) · sameday 민감도(sensitivity)

실행: python scripts/box_validation.py [--sweep] [--gap 0.07] [--no-push] [--out P]
      (--sweep: 갭 스윕 → data/box_validation_sweep.json)
"""
import argparse
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
SWEEP = REPO / "data" / "box_validation_sweep.json"
PIT = Path("/home/lucky/backtest/data/pit")
MASTER = PIT / "master.pkl"

UNIT, FEE = 100_000, 0.002
K = 20                 # 동시 보유 슬롯 (진입 = 순위 상위 20)
GAP = 0.07             # 확정 갭 (C 구성 — 12년 8갭 스윕 최적)
TS = 15                # 시간중단 영업일 (확정 — 5/6/7% 전 구성 개선)
MA200 = True           # 200일선(40주 완성 주봉) 하회 진입 제외 (확정 — C안)
AMT_MIN = 1e9
START = 20140901       # 스캔 시작일 (fdr 과거 상한 2014-08)
COVER = 0.375          # 사다리 커버 깊이 → rungs = max(4, round(COVER/gap)) ≈ 하단 28~30%
SWEEP_GAPS = [0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]

SAMEDAY_KO = {"all": "낙관(저가→고가 가정)", "close": "현실(종가≥목표가 때만)",
              "none": "보수(당일 신규차수 매도 금지)"}
EXIT_KO = {"close0": "종가 확정 즉시 청산", "close5": "종가 -5% 버퍼",
           "low0": "장중 하단 터치 청산", "low5": "장중 -5% 청산",
           "box": "박스 소멸 시만 (구 규칙)"}
TIMING_KO = {"real": "실전(진입·청산 모두 다음 영업일 시가)",
             "close": "단순 시뮬(신호일 종가 진입·확정 종가 청산)"}
DEEP_MAPS = None      # lazy — fetch_deep_maps()


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
        if drift is not None:
            if drift > 0:
                s += 1
            elif drift >= -0.05:
                s -= 1
            elif drift >= -0.10:
                s -= 2
            else:
                s -= 3
        if ratio is not None and ratio <= 1.8:
            s += 1
        return s


def fetch_deep_maps():
    """지수 딥베어(200일MA×0.93 하회) 맵 — KS11/KQ11 (옵션용, 기본 미사용)."""
    global DEEP_MAPS
    if DEEP_MAPS is not None:
        return DEEP_MAPS
    maps = {}
    try:
        import FinanceDataReader as fdr
        for sym, mkt in (("KS11", "KOSPI"), ("KQ11", "KOSDAQ")):
            df = fdr.DataReader(sym, "2013-01-01")
            c = df["Close"].astype(float)
            maps[mkt] = {
                "dn": np.array([int(d.strftime("%Y%m%d")) for d in df.index],
                               dtype=np.int64),
                "deep": np.nan_to_num(
                    (c < c.rolling(200).mean() * 0.93).values.astype(float))}
    except Exception:
        maps = {}
    DEEP_MAPS = maps
    return DEEP_MAPS


def load_universe():
    """종목 배열 + 트리거 후보 (매수존·유동성·박스지속·비이탈일) + 주봉 40주 MA 상태."""
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
        o = df["Open"].values.astype(float)
        v = df["Volume"].values.astype(float)
        dn = np.array([int(d.strftime("%Y%m%d")) for d in df.index],
                      dtype=np.int64)
        hi60 = pd.Series(h).rolling(60).max().values
        lo60 = pd.Series(l).rolling(60).min().values
        drift_raw = np.full(n, np.nan)
        drift_raw[60:] = c[60:] / c[:-60] - 1
        drift = pd.Series(drift_raw).rolling(5).mean().values   # 5일 스무딩 (box_alert 동기)
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
        # 전일 하단 (이탈 판정·진입 금지 체크용)
        lo60_prev = np.full(n, np.nan)
        lo60_prev[1:] = lo60[:-1]
        trig = (persist & (posf <= 0.5) & (np.nan_to_num(amt20) >= AMT_MIN)
                & (c >= lo60_prev)          # 이탈일(종가<전일하단)에는 진입 금지
                & (dn >= START))
        ti = np.where(trig)[0]
        if not len(ti):
            continue
        sc = np.array([priority_of(int(pdays[i]), float(posf[i]),
                                   float(atr[i]) if not np.isnan(atr[i]) else None,
                                   float(drift[i]) if not np.isnan(drift[i]) else 0.0,
                                   float(ratio[i]) if not np.isnan(ratio[i]) else None)
                       for i in ti], dtype=np.int64)
        # 주봉 40주(≈200일) MA 상태 — 신호일 기준 마지막 완성 주봉 (HTS 주봉 시점 동일)
        wk = pd.Series(c, index=df.index).resample("W-FRI").last().dropna()
        wlab = np.array([int(d.strftime("%Y%m%d")) for d in wk.index],
                        dtype=np.int64)
        wc = wk.values.astype(float)
        ta40 = np.ones(len(ti), dtype=float)
        for t, j in enumerate(ti):
            k = int(np.searchsorted(wlab, int(dn[j]), side="right")) - 1
            if k >= 39:
                ta40[t] = 1.0 if wc[k] > wc[k - 39:k + 1].mean() else 0.0
        U.append({"code": code, "name": name, "mkt": str(r.market),
                  "dn": dn, "o": o, "h": h, "l": l, "c": c,
                  "persist": persist, "lo60": lo60, "hi60": hi60,
                  "lo60_prev": lo60_prev, "drift": drift, "ratio": ratio,
                  "amt20": amt20, "tatr": np.nan_to_num(atr[ti]),
                  "trig": ti, "sc": sc, "ta40": ta40})
    return U


def vanish_reason(s, j):
    """박스 소멸 당일 조건값으로 사유 분류."""
    rt = s["ratio"][j] if not np.isnan(s["ratio"][j]) else None
    dr = s["drift"][j] if not np.isnan(s["drift"][j]) else None
    am = s["amt20"][j] if not np.isnan(s["amt20"][j]) else None
    if rt is not None and rt > 2.2:
        return "상방확장"
    if rt is not None and rt < 1.25:
        return "범위축소"
    if dr is not None and abs(dr) > 0.20:
        return "추세발생"
    if am is not None and am < AMT_MIN:
        return "유동성미달"
    return "기타"


def run(gap, rungs, exit_rule, sameday, timing, ts, use_ma200, use_deep,
        U, all_days, ev_days, detail=False):
    """한 변형 실행. 상세(detail=True)면 사이트 렌더용 전체 블록 반환."""
    entry_open = (timing == "real")
    exit_open = (timing == "real")
    if use_deep:
        deep_maps = fetch_deep_maps()

    def deep_on(mkt, D):
        m = deep_maps.get(mkt) or deep_maps.get("KOSPI")
        i = int(np.searchsorted(m["dn"], D, side="right")) - 1
        return bool(m["deep"][i]) if i >= 0 else False

    holdings = {}
    done, curve = [], []
    cap_daily = []
    max_pos = max_rungs = 0

    def exit_px(s, j):
        """청산 체결가 — real: 다음 영업일 시가 / close: 현재 종가."""
        if exit_open and j + 1 < len(s["dn"]):
            return float(s["o"][j + 1]), int(s["dn"][j + 1])
        return float(s["c"][j]), int(s["dn"][j])

    def close_position(done, holdings, curve, i, px, kind, day):
        hst = holdings.pop(i)
        if hst.get("pending"):       # 미체결 진입 — 기록 없음
            return
        for k in range(len(hst["levels"])):
            if hst["held"][k]:
                hst["net"] += (px - hst["levels"][k]) / hst["levels"][k] * UNIT
                if k == 0 and hst["grid_n"] == 1:
                    hst["loss0"] = True
        done.append({"code": i, "entry_day": hst["entry_day"],
                     "exit_day": int(day), "score": hst["score"],
                     "escape0": hst["escape0"], "loss0": hst["loss0"],
                     "cycles": hst["cycles"], "net": hst["net"],
                     "days": int(hst["j"] - hst["j_entry"]), "kind": kind})
        curve.append((int(day), hst["net"]))

    for D in all_days:
        if D < START:
            continue
        for i in list(holdings):
            hst, s = holdings[i], U[i]
            dn = s["dn"]
            j = int(np.searchsorted(dn, D))
            if j >= len(dn) or dn[j] != D:
                # 데이터갭 — 재개봉 시가 / 종가 청산
                if hst.get("pending"):
                    continue
                if exit_open and j < len(dn):
                    px, day = float(s["o"][j]), int(dn[j])
                else:
                    px, day = float(s["c"][hst["j"]]), int(D)
                close_position(done, holdings, curve, i, px, "데이터갭", day)
                continue
            lo_prev = float(s["lo60_prev"][j])
            px_c = float(s["c"][j])
            lw = float(s["l"][j])
            # 진입 실행 (real: 전일 신호 → 오늘 시가 체결)
            if hst.get("pending"):
                hst["pending"] = False
                epx = float(s["o"][j])
                hst["entry"] = epx
                hst["levels"] = [epx * (1 - gap) ** k for k in range(rungs)]
                hst["held"] = [False] * rungs
                hst["held"][0] = True
                hst["j_entry"] = j
                hst["j"] = j
            # 1) 시간중단
            if ts > 0 and (hst["j"] - hst["j_entry"]) >= ts:
                px, day = exit_px(s, j)
                close_position(done, holdings, curve, i, px,
                               f"시간중단{ts}일", day)
                continue
            # 2) 청산 규칙 (전일 박스 경계 기준)
            if exit_rule == "close0" and px_c < lo_prev:
                px, day = exit_px(s, j)
                close_position(done, holdings, curve, i, px, "이탈즉시(종가)", day)
                continue
            if exit_rule == "close5" and px_c < lo_prev * 0.95:
                px, day = exit_px(s, j)
                close_position(done, holdings, curve, i, px, "이탈-5%(종가)", day)
                continue
            if exit_rule == "low0" and lw < lo_prev:
                px, day = exit_px(s, j)
                close_position(done, holdings, curve, i, px, "이탈즉시(장중)", day)
                continue
            if exit_rule == "low5" and lw < lo_prev * 0.95:
                px, day = exit_px(s, j)
                close_position(done, holdings, curve, i, px, "이탈-5%(장중)", day)
                continue
            # 3) 박스 소멸
            if not bool(s["persist"][j]):
                rsn = vanish_reason(s, j)
                kind = ("상방돌파 소멸" if rsn == "상방확장" else "박스소멸")
                px, day = exit_px(s, j)
                close_position(done, holdings, curve, i, px, kind, day)
                continue
            hst["j"] = j
            # 4) 하드스톱 백스톱 (-15%)
            stop_px = lo_prev * 0.85
            if lw <= stop_px:
                close_position(done, holdings, curve, i, stop_px, "손절-15%", int(dn[j]))
                continue
            # 5) 그리드 진행 — 저가 채결(매수) → 고가 채결(매도)
            levels, held = hst["levels"], hst["held"]
            bought = set()
            for k in range(1, rungs):
                if not held[k] and lw <= levels[k]:
                    held[k] = True
                    bought.add(k)
            for k in range(rungs):
                if not held[k]:
                    continue
                tgt = levels[k] * (1 + gap)
                if float(s["h"][j]) < tgt:
                    continue
                if (k in bought) and sameday != "all":
                    if sameday == "none":
                        continue
                    if sameday == "close" and px_c < tgt:
                        continue
                hst["net"] += gap * UNIT - FEE * UNIT
                held[k] = False
                hst["cycles"] += 1
                if k == 0 and hst["grid_n"] == 1:
                    hst["escape0"] = True
            if not any(held):      # 재눌림 → 재그리드
                pos = ((px_c - float(s["lo60"][j]))
                       / max(float(s["hi60"][j]) - float(s["lo60"][j]), 1e-9))
                if pos <= 0.5:
                    hst["grid_n"] += 1
                    hst["levels"] = [px_c * (1 - gap) ** k for k in range(rungs)]
                    hst["held"] = [False] * rungs
                    hst["held"][0] = True
            held_cnt = sum(held)
            max_rungs = max(max_rungs, held_cnt)
        cap_daily.append(sum(sum(h["held"]) for h in holdings.values()) * UNIT)
        max_pos = max(max_pos, len(holdings))
        free = K - len(holdings)
        if free > 0:
            for d, nsc, i, j in ev_days.get(int(D), []):
                if free <= 0:
                    break
                if i in holdings:
                    continue
                s = U[i]
                if use_deep and deep_on(s["mkt"], int(D)):
                    continue
                if use_ma200:
                    tj = int(np.searchsorted(s["trig"], j))
                    if s["ta40"][tj] != 1.0:
                        continue
                holdings[i] = {"j": j, "j_entry": j, "entry_day": int(D),
                               "entry": float(s["c"][j]), "score": -nsc,
                               "levels": [], "held": [], "grid_n": 1,
                               "escape0": False, "loss0": False,
                               "cycles": 0, "net": 0.0,
                               "pending": entry_open}
                if not entry_open:
                    epx = float(s["c"][j])
                    holdings[i]["entry"] = epx
                    holdings[i]["levels"] = [epx * (1 - gap) ** k
                                             for k in range(rungs)]
                    holdings[i]["held"] = [False] * rungs
                    holdings[i]["held"][0] = True
                free -= 1
    for i in list(holdings):
        s = U[i]
        hst = holdings[i]
        if hst.get("pending"):
            holdings.pop(i)
            continue
        close_position(done, holdings, curve, i,
                       float(s["c"][hst["j"]]), "데이터종료",
                       int(s["dn"][hst["j"]]))

    y0 = (min((p["entry_day"] for p in done), default=START) // 10000)
    y1 = (max((p["exit_day"] for p in done), default=START) // 10000)
    years = max(y1 - y0 + 1, 1)
    nets = [p["net"] for p in done]
    esc = sum(1 for p in done if p["escape0"]) / max(len(done), 1) * 100
    win = sum(1 for x in nets if x > 0) / max(len(nets), 1) * 100
    eq = peak = dd = 0.0
    for d, v in sorted(curve):
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
    av_cap = float(np.mean(cap_daily)) if cap_daily else 0.0
    pk_cap = float(np.max(cap_daily)) if cap_daily else 0.0
    p95_cap = float(np.percentile(cap_daily, 95)) if cap_daily else 0.0
    reserve = K * rungs * UNIT
    summary = {
        "positions": len(done), "years": years,
        "start_year": y0, "end_year": y1,
        "turnover_year": round(len(done) / years, 1),
        "turnover_slot": round(len(done) / years / K, 2),
        "escape_rate": round(esc, 1),
        "loss0_rate": round(sum(1 for p in done if p["loss0"])
                            / max(len(done), 1) * 100, 1),
        "win_rate": round(win, 1),
        "hold_avg": round(sum(p["days"] for p in done) / max(len(done), 1), 1),
        "hold_median": int(sorted(p["days"] for p in done)[len(done) // 2]) if done else 0,
        "cycles_avg": round(sum(p["cycles"] for p in done) / max(len(done), 1), 1),
        "net_total": round(sum(nets)),
        "mdd_won": round(dd),
        "mdd_vs_capital_pct": round(dd / max(av_cap, 1) * 100, 1),
        "loss_years": sum(1 for v in yy.values() if v["net"] < 0),
        "n_years": len(yy),
    }
    res = {"summary": summary}
    if not detail:
        res.update({"avg_capital": round(av_cap), "peak_capital": round(pk_cap),
                    "p95_capital": round(p95_cap),
                    "ret_capital_pct": round(summary["net_total"]
                                             / max(av_cap, 1) / years * 100, 1),
                    "ret_reserve_pct": round(summary["net_total"]
                                             / max(reserve, 1) / years * 100, 2),
                    "max_positions": max_pos, "max_rungs": max_rungs})
        return res
    # 상세 (사이트 렌더용)
    bk = {"0~2": [], "3~4": [], "5~6": [], "7+": []}
    for p in done:
        sc = p["score"]
        key = "7+" if sc >= 7 else ("5~6" if sc >= 5 else ("3~4" if sc >= 3 else "0~2"))
        bk[key].append(p)
    score_escape = {
        kk: {"n": len(v),
             "esc_rate": round(sum(1 for p in v if p["escape0"]) / len(v) * 100, 1) if v else None,
             "avg_net": round(sum(p["net"] for p in v) / len(v)) if v else None}
        for kk, v in bk.items()}
    code_map = {i: s["code"] for i, s in enumerate(U)}
    sh = defaultdict(lambda: {"ep": 0, "esc": 0, "days": 0, "net": 0.0})
    for p in done:
        tc = code_map.get(p["code"], "")
        if not tc:
            continue
        sh[tc]["ep"] += 1
        if p["escape0"]:
            sh[tc]["esc"] += 1
        sh[tc]["days"] += p["days"]
        sh[tc]["net"] += p["net"]
    stock_history = {
        tc: {"ep": d["ep"], "esc": d["esc"],
             "hold": round(d["days"] / d["ep"], 1) if d["ep"] else None,
             "net": round(d["net"] / d["ep"]) if d["ep"] else 0}
        for tc, d in sh.items()}
    name_of = {i: s["name"] for i, s in enumerate(U)}
    recent = sorted(done, key=lambda p: p["exit_day"], reverse=True)[:30]
    res.update({
        "yearly": [{"year": y, "net": round(v["net"]), "positions": v["n"],
                    "esc_rate": round(v["esc"] / v["n"] * 100, 1) if v["n"] else None}
                   for y, v in sorted(yy.items())],
        "exit_kinds": dict(kinds),
        "score_escape": score_escape,
        "capital": {"avg": round(av_cap), "peak": round(pk_cap),
                    "p95": round(p95_cap), "reserve": round(reserve),
                    "ret_capital_pct": round(summary["net_total"]
                                             / max(av_cap, 1) / years * 100, 1),
                    "ret_reserve_pct": round(summary["net_total"]
                                             / max(reserve, 1) / years * 100, 2),
                    "max_positions": max_pos, "max_rungs": max_rungs},
        "recent_positions": [
            {"name": name_of.get(p["code"], "?"), "entry_day": str(p["entry_day"]),
             "exit_day": str(p["exit_day"]), "days": p["days"], "score": p["score"],
             "escaped": p["escape0"], "cycles": p["cycles"], "net": round(p["net"]),
             "kind": p["kind"]} for p in recent],
        "stock_history": stock_history,
    })
    return res


def main():
    ap = argparse.ArgumentParser(description="박스 Top-20 검증 시뮬레이션 (확정 구성 C)")
    ap.add_argument("--sameday", choices=["close", "none", "all"], default="close")
    ap.add_argument("--exit", choices=["close0", "close5", "low0", "low5", "box"],
                    default="close0")
    ap.add_argument("--gap", type=float, default=GAP)
    ap.add_argument("--rungs", type=int, default=0)
    ap.add_argument("--ts", type=int, default=TS)
    ap.add_argument("--timing", choices=["real", "close"], default="real")
    ap.add_argument("--no-ma200", action="store_true")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    rungs = args.rungs if args.rungs else max(4, int(round(COVER / args.gap)))
    use_ma200 = not args.no_ma200

    t0 = tmod.time()
    U = load_universe()
    print(f"[INFO] 유니버스 {len(U):,}종목 ({tmod.time()-t0:.0f}s)", flush=True)
    all_days = np.sort(np.unique(np.concatenate([s["dn"] for s in U])))
    ev = []
    for i, s in enumerate(U):
        for j, sc in zip(s["trig"], s["sc"]):
            ev.append((int(s["dn"][j]), -int(sc), i, int(j)))
    ev.sort()
    ev_days = defaultdict(list)
    for e in ev:
        ev_days[e[0]].append(e)

    if args.sweep:
        rows = []
        for gp in SWEEP_GAPS:
            rg = max(4, int(round(COVER / gp)))
            for mode in ("close", "none", "all"):
                r = run(gp, rg, args.exit, mode, args.timing, args.ts,
                        use_ma200, args.deep, U, all_days, ev_days)
                sm = r["summary"]
                rows.append({"gap_pct": gp * 100, "sameday": mode, "rungs": rg,
                             **sm, **{kk: r[kk] for kk in
                                      ("avg_capital", "peak_capital",
                                       "p95_capital", "ret_capital_pct",
                                       "ret_reserve_pct", "max_rungs")}})
                print(f"  갭 {gp*100:>4.1f}% · {mode:<5} → 총 {sm['net_total']:+,}원"
                      f" · MDD {sm['mdd_won']:+,} · 투입수익 {r['ret_capital_pct']:+.1f}%/년",
                      flush=True)
        SWEEP.write_text(json.dumps(
            {"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
             "exit": args.exit, "ts": args.ts, "ma200": use_ma200,
             "timing": args.timing, "rows": rows}, ensure_ascii=False),
            encoding="utf-8")
        if not args.no_push:
            subprocess.run(["git", "-C", str(REPO), "add", "-A"], capture_output=True)
            subprocess.run(["git", "-C", str(REPO), "commit", "-m",
                            "chore: 갭 스윕 갱신"], capture_output=True)
            subprocess.run(["git", "-C", str(REPO), "push"], capture_output=True)
        print(f"[INFO] 스윕 완료 → {SWEEP.name}")
        return

    r = run(args.gap, rungs, args.exit, args.sameday, args.timing, args.ts,
            use_ma200, args.deep, U, all_days, ev_days, detail=True)
    sens = {}
    for m in ("all", "close", "none"):
        if m == args.sameday:
            continue
        r2 = run(args.gap, rungs, args.exit, m, args.timing, args.ts,
                 use_ma200, args.deep, U, all_days, ev_days)
        sens[m] = {"net_total": r2["summary"]["net_total"],
                   "mdd_won": r2["summary"]["mdd_won"],
                   "escape_rate": r2["summary"]["escape_rate"],
                   "turnover_year": r2["summary"]["turnover_year"]}
    s = r["summary"]
    out = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": {"slots": K, "buy_zone": "위치≤50%",
                   "unit": f"{UNIT//10000}만/차",
                   "gap": f"{args.gap*100:g}% 고정",
                   "start": str(s["start_year"]),
                   "end": str(s["end_year"])},
        "assumptions": {
            "timing": args.timing, "timing_ko": TIMING_KO[args.timing],
            "sameday": args.sameday, "sameday_ko": SAMEDAY_KO[args.sameday],
            "exit": args.exit, "exit_ko": EXIT_KO[args.exit],
            "ts": args.ts, "ts_ko": f"시간중단 {args.ts} 영업일 진입 금지" if args.ts > 0 else "시간중단 없음",
            "ma200": use_ma200,
            "ma200_ko": "200일선(40주 완성 주봉) 하회 진입 제외" if use_ma200 else "200일선 필터 없음",
            "deep": args.deep,
            "deep_ko": "지수 딥베어(200일MA×93%) 진입 금지" if args.deep else "지수 필터 없음",
            "gap_pct": args.gap * 100, "rungs": rungs,
            "unit": UNIT, "fee_roundtrip": FEE,
            "entry": "신호일 종가 알림 → 다음 영업일 시가 체결 (이탈일 진입 금지)",
            "stop": "전일 박스 하단 -15% 백스톱",
            "escape_def": "탈출율 = 첫 차수(진입)가 +갭 가격에 매도된 비율"},
        "sensitivity": {m: {"sameday_ko": SAMEDAY_KO[m], **v} for m, v in sens.items()},
        **{kk: r[kk] for kk in ("summary", "yearly", "exit_kinds", "score_escape",
                                "capital", "recent_positions", "stock_history")},
    }
    print(f"[INFO] 포지션 {s['positions']:,} ({s['start_year']}~{s['end_year']}"
          f" {s['years']}년) · 탈출율 {s['escape_rate']}% · 지속 {s['hold_avg']}일 · "
          f"총 {s['net_total']:+,}원 · MDD {s['mdd_won']:+,}원"
          f"({s['mdd_vs_capital_pct']}%) · 적자연도 {s['loss_years']}/{s['n_years']} · "
          f"평균자본 {r['capital']['avg']:,}원 · 투입수익 {r['capital']['ret_capital_pct']:+.1f}%/년"
          f" · 예약수익 {r['capital']['ret_reserve_pct']:+.2f}%/년",
          flush=True)
    dest_out = Path(args.out) if args.out else OUT
    dest_out.parent.mkdir(parents=True, exist_ok=True)
    dest_out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    if args.no_push or args.out:
        print(f"[INFO] 저장 {dest_out} (push 생략)")
        return
    subprocess.run(["git", "-C", str(REPO), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(REPO), "commit", "-m",
                    f"chore: 시뮬 갱신 (포지션 {s['positions']})"], capture_output=True)
    subprocess.run(["git", "-C", str(REPO), "push"], capture_output=True)
    print("[INFO] git push 완료")


if __name__ == "__main__":
    main()