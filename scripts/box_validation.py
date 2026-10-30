#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""박스권 Top-20(매수존) 순위 진입 시뮬레이션 — 검증 탭 데이터 (box-monitor 전용)
============================================================================
매일: LIVE 박스(60일 높이 1.25~2.2 · 추세≤20% · ATR≥3% · 20일 연속 · 거래대금
10억+) 중 **매수존(위치≤50%)**만 후보 → priority_score 랭킹 → **상위 20 슬롯** 진입.
포지션: 그리드 사다리 (진입 종가 앵커·갭 고정·러그매도 되팔기·재그리드 허용
· 박스 소멸·데이터갭 시 청산) — 슬롯 해제 후 다음 날 최고 순위 진입.

■ 가정 스위치 (Phase 0 — 2026-10-30)
   --sameday close|none|all   당일 신규체결 차수의 매도 가정
        all  : 저가→고가 순서 가정 (상한선 — 현행 시뮬의 낙관)
        close: 종가 ≥ 목표가일 때만 당일 매도 허용 (지정가 사다리 운용의 현실 기준)
        none : 당일 신규차수 매도 금지 (보수 — 하한선)
   --exit close0|close5|low0|low5|box
        close0(기본): 종가 < 전일 박스 하단 → 그 종가에 청산 (확정 규칙)
        close5      : 종가 < 전일 하단×0.95
        low0        : 저가 < 전일 하단 → 하단가 청산
        low5        : 저가 < 전일 하단×0.95 → ×0.95 청산
        box         : 박스 소멸 시만 청산 (구 규칙)
   하드스톱 -15%(전일 하단 기준)은 모든 규칙의 백스톱.
   진입: 신호일 종가. 단, 종가 < 전일 하단(이탈일)에는 진입 금지.

■ 산출: 회전 · 탈출율(첫 차수 +갭 매도 비율) · 지속 · 연도별 손익 · 승률 · MDD
   · 청산사유(방향 분리) · 점수구간별 탈출율(score_escape) · 투입자본/자본수익률
   · 가정 명시(assumptions) · 다른 sameday 모드 민감도(sensitivity)

실행: python scripts/box_validation.py [--sweep] [--gap 0.025] [--no-push] [--out P]
      (--sweep: 갭 2.5~10% × sameday close/none/all 비교 → data/box_validation_sweep.json)
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
GAP = 0.025            # 기본 갭 (결정 전까지 유지)
AMT_MIN = 1e9
START = 20140901       # 스캔 시작일 (fdr 과거 상한 2014-08)
COVER = 0.375          # 사다리 커버 깊이 → rungs = max(4, round(COVER/gap)) ≈ 하단 28~30%
SWEEP_GAPS = [0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]

SAMEDAY_KO = {"all": "낙관(저가→고가 가정)", "close": "현실(종가≥목표가 때만)",
              "none": "보수(당일 신규차수 매도 금지)"}
EXIT_KO = {"close0": "종가 확정 즉시 청산", "close5": "종가 -5% 버퍼",
           "low0": "장중 하단 터치 청산", "low5": "장중 -5% 청산",
           "box": "박스 소멸 시만 (구 규칙)"}


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


def load_universe():
    """종목 배열 + 트리거 후보 (매수존·유동성·박스지속·비이탈일)."""
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
        kind_map = {int(j): "std" for j in np.where(trig)[0]}
        # ── W-box (주봉 종가 밴드): 13주 종가가 ±30% 밴드 · 순드리프트 ≤15%
        #    진입: 주봉박스 하단부(밴드 하위 절반) · 소멸: 종가가 밴드 이탈
        wk = pd.Series(c, index=df.index).resample("W-FRI").last().dropna()
        wclose = wk.values.astype(float)
        wend = np.array([int(d.strftime("%Y%m%d")) for d in wk.index],
                        dtype=np.int64)
        nw = len(wclose)
        wflagw = np.zeros(nw, dtype=bool)
        wlo_w = np.zeros(nw)
        whi_w = np.zeros(nw)
        for k in range(12, nw):
            seg = wclose[k - 12:k + 1]
            wlo_w[k], whi_w[k] = seg.min(), seg.max()
            wflagw[k] = (seg.max() / max(seg.min(), 1e-9) - 1 <= 0.30
                         and abs(seg[-1] / max(seg[0], 1e-9) - 1) <= 0.15)
        kk = np.searchsorted(wend, dn) - 1
        okw = kk >= 12
        wflag = np.where(okw, wflagw[np.clip(kk, 0, nw - 1)], False)
        wlo = np.where(okw, wlo_w[np.clip(kk, 0, nw - 1)], 0.0)
        whi = np.where(okw, whi_w[np.clip(kk, 0, nw - 1)], 0.0)
        wpos = np.where(np.maximum(whi - wlo, 1e-9) > 0,
                        (c - wlo) / np.maximum(whi - wlo, 1e-9), 1.0)
        liq = np.nan_to_num(amt20) >= AMT_MIN
        notbreak = np.nan_to_num(lo60_prev, nan=1e18) <= c
        wdays = np.where(wflag & (wpos <= 0.5) & liq & notbreak
                         & (dn >= START))[0]
        for j in wdays:
            kind_map.setdefault(int(j), "wbox")
        # ── A-box (상승 저점 추세선): 피벗 저가 3개 연속 상승 + 종가 ≥ 선×0.995
        #    진입: 추세선 근접(종가 ≤ 선×1.05) · 소멸: 종가 < 선×0.995 또는 구조 붕괴
        pmin = pd.Series(l).rolling(11, center=True).min().values
        PidA, PvA, PcA = [], [], []
        for i in range(5, n - 5):
            if l[i] == pmin[i] and l[i] < l[i - 1] and l[i] < l[i + 1]:
                PidA.append(i)
                PvA.append(float(l[i]))
                PcA.append(i + 5)
        PidA = np.array(PidA, dtype=np.int64)
        PvA = np.array(PvA)
        PcA = np.array(PcA, dtype=np.int64)
        abox_ok = np.zeros(n, dtype=bool)
        abox_rise = np.zeros(n, dtype=bool)
        tlv = np.full(n, np.nan)
        if len(PvA) >= 3:
            kc = np.searchsorted(PcA, np.arange(n), side="right") - 1
            okk = kc >= 2
            i1 = PidA[np.clip(kc - 1, 0, None)]
            i2 = PidA[np.clip(kc, 0, None)]
            v1 = PvA[np.clip(kc - 1, 0, None)]
            v2 = PvA[np.clip(kc, 0, None)]
            v0 = PvA[np.clip(kc - 2, 0, None)]
            rising = (v2 > v1 * 1.005) & (v1 > v0 * 0.995)
            slope = (v2 - v1) / np.maximum(i2 - i1, 1)
            tl_now = v2 + slope * (np.arange(n) - i2)
            ok = okk & rising & (tl_now > 0) & (c >= tl_now * 0.995)
            abox_ok = ok & okk
            abox_rise = okk & rising
            tlv = np.where(ok, tl_now, np.nan)
        atrig = (abox_ok & (c <= np.nan_to_num(tlv, nan=0) * 1.05)
                 & (np.nan_to_num(tlv, nan=0) > 0)
                 & liq & (dn >= START))
        for j in np.where(atrig)[0]:
            kind_map.setdefault(int(j), "abox")
        ti = np.array(sorted(kind_map), dtype=np.int64)
        kinds_arr = [kind_map[int(j)] for j in ti]
        if not len(ti):
            continue
        sc = np.array([priority_of(
                           int(pdays[j]) if kind_map[int(j)] == "std" else 0,
                           float(posf[j]),
                           float(atr[j]) if not np.isnan(atr[j]) else None,
                           float(drift[j]) if not np.isnan(drift[j]) else 0.0,
                           float(ratio[j]) if not np.isnan(ratio[j]) else None)
                       for j in ti], dtype=np.int64)
        U.append({"code": code, "name": name, "dn": dn, "h": h, "l": l,
                  "c": c, "atr": atr, "persist": persist, "lo60": lo60,
                  "hi60": hi60, "lo60_prev": lo60_prev, "posf": posf,
                  "drift": drift, "ratio": ratio, "amt20": amt20,
                  "wflag": wflag, "wlo": wlo, "whi": whi, "wpos": wpos,
                  "abox_ok": abox_ok, "abox_rise": abox_rise, "tlv": tlv,
                  "trig": ti, "kinds": kinds_arr, "sc": sc})
    return U


def vanish_reason(s, j):
    """박스 소멸 당일 조건값으로 사유 분류."""
    rt = s["ratio"][j] if not np.isnan(s["ratio"][j]) else None
    dr = s["drift"][j] if not np.isnan(s["drift"][j]) else None
    at = s["atr"][j] if not np.isnan(s["atr"][j]) else None
    am = s["amt20"][j] if not np.isnan(s["amt20"][j]) else None
    if rt is not None and rt > 2.2:
        return "상방확장"
    if rt is not None and rt < 1.25:
        return "범위축소"
    if dr is not None and abs(dr) > 0.20:
        return "추세발생"
    if at is not None and at < 0.030:
        return "변동성축소"
    if am is not None and am < AMT_MIN:
        return "유동성미달"
    return "기타"


def run(gap, rungs, exit_rule, sameday, use_ext, U, all_days, ev_days,
        detail=False):
    """한 변형 실행. 상세(detail=True)면 사이트 렌더용 전체 블록 반환."""
    holdings = {}
    done, curve = [], []
    cap_daily, max_pos, max_rungs = [], 0, 0
    K_ = K

    def close_position(done, holdings, curve, i, px, why, day):
        hst = holdings.pop(i)
        for k in range(rungs):
            if hst["held"][k]:
                hst["net"] += (px - hst["levels"][k]) / hst["levels"][k] * UNIT
                if k == 0 and hst["grid_n"] == 1:
                    hst["loss0"] = True
        done.append({"code": i, "entry_day": hst["entry_day"],
                     "exit_day": int(day), "score": hst["score"],
                     "escape0": hst["escape0"], "loss0": hst["loss0"],
                     "cycles": hst["cycles"], "net": hst["net"],
                     "days": hst["j"] - hst["j_entry"], "kind": why})
        curve.append((int(day), hst["net"]))

    for D in all_days:
        if D < START:
            continue
        for i in list(holdings):
            hst, s = holdings[i], U[i]
            dn = s["dn"]
            j = int(np.searchsorted(dn, D))
            if j >= len(dn) or dn[j] != D:
                close_position(done, holdings, curve, i,
                               float(s["c"][hst["j"]]), "데이터갭", D)
                continue
            knd = hst.get("kind", "std")
            lo_prev = float(np.nan_to_num(s["lo60_prev"][j], nan=0.0))
            wlo = float(s["wlo"][j])
            whi = float(s["whi"][j])
            px, lw = float(s["c"][j]), float(s["l"][j])
            # 1) 청산 규칙 (전일 박스 경계 기준) — std·wbox만 60일 경계 적용
            if knd in ("std", "wbox"):
                if exit_rule == "close0" and px < lo_prev:
                    close_position(done, holdings, curve, i, px,
                                   "이탈즉시(종가)", D)
                    continue
                if exit_rule == "close5" and px < lo_prev * 0.95:
                    close_position(done, holdings, curve, i, px,
                                   "이탈-5%(종가)", D)
                    continue
                if exit_rule == "low0" and lw < lo_prev:
                    close_position(done, holdings, curve, i, lo_prev,
                                   "이탈즉시(장중)", D)
                    continue
                if exit_rule == "low5" and lw < lo_prev * 0.95:
                    close_position(done, holdings, curve, i, lo_prev * 0.95,
                                   "이탈-5%(장중)", D)
                    continue
            # 2A) 박스 소멸 — std: 60일 조건 / wbox: 주봉 밴드 / abox: 추세선·구조
            if knd == "std" and not bool(s["persist"][j]):
                rsn = vanish_reason(s, j)
                why = ("상방돌파 소멸" if rsn == "상방확장" else "박스소멸")
                close_position(done, holdings, curve, i, px, why, D)
                continue
            if knd == "wbox" and (not bool(s["wflag"][j])
                                  or px < wlo or px > whi):
                close_position(done, holdings, curve, i, px, "주봉박스 소멸", D)
                continue
            if knd == "abox" and not bool(s["abox_ok"][j]):
                why = ("추세선 붕괴" if not bool(s["abox_rise"][j])
                       else "추세선 이탈")
                close_position(done, holdings, curve, i, px, why, D)
                continue
            hst["j"] = j
            # 3) 하드스톱 백스톱 (-15%) — 전일 60일 하단 기준
            stop_px = lo_prev * 0.85
            if lo_prev > 0 and lw <= stop_px:
                close_position(done, holdings, curve, i, stop_px, "손절-15%", D)
                continue
            # 4) 그리드 진행 — 저가 채결(매수) → 고가 채결(매도)
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
                    if sameday == "close" and px < tgt:
                        continue
                hst["net"] += gap * UNIT - FEE * UNIT
                held[k] = False
                hst["cycles"] += 1
                if k == 0 and hst["grid_n"] == 1:
                    hst["escape0"] = True
            if not any(held):      # 재눌림 → 재그리드
                pos = ((px - float(s["lo60"][j]))
                       / max(float(s["hi60"][j]) - float(s["lo60"][j]), 1e-9))
                if pos <= 0.5:
                    hst["grid_n"] += 1
                    hst["levels"] = [px * (1 - gap) ** k for k in range(rungs)]
                    hst["held"] = [False] * rungs
                    hst["held"][0] = True
            held_cnt = sum(holdings[i]["held"])
            max_rungs = max(max_rungs, held_cnt)
        cap_daily.append(sum(
            sum(h["held"]) for h in holdings.values()) * UNIT)
        max_pos = max(max_pos, len(holdings))
        free = K_ - len(holdings)
        if free > 0:
            for d, nsc, i, j, knd in ev_days.get(int(D), []):
                if free <= 0:
                    break
                if i in holdings:
                    continue
                if (not use_ext) and knd != "std":
                    continue
                s = U[i]
                holdings[i] = {"j": j, "j_entry": j, "entry_day": int(D),
                               "entry": float(s["c"][j]), "score": -nsc,
                               "gap": gap, "kind": knd,
                               "levels": [float(s["c"][j]) * (1 - gap) ** k
                                          for k in range(rungs)],
                               "held": [False] * rungs,
                               "grid_n": 1, "escape0": False, "loss0": False,
                               "cycles": 0, "net": 0.0}
                holdings[i]["held"][0] = True
                free -= 1
    for i in list(holdings):
        s = U[i]
        close_position(done, holdings, curve, i, float(s["c"][holdings[i]["j"]]),
                       "데이터종료", int(all_days[-1]))

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
    summary = {
        "positions": len(done), "years": years,
        "start_year": y0, "end_year": y1,
        "turnover_year": round(len(done) / years, 1),
        "turnover_slot": round(len(done) / years / K_, 2),
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
                    "p95": round(p95_cap),
                    "ret_capital_pct": round(summary["net_total"]
                                             / max(av_cap, 1) / years * 100, 1),
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
    ap = argparse.ArgumentParser(description="박스 Top-20 검증 시뮬레이션")
    ap.add_argument("--sameday", choices=["close", "none", "all"], default="close")
    ap.add_argument("--exit", choices=["close0", "close5", "low0", "low5", "box"],
                    default="close0")
    ap.add_argument("--gap", type=float, default=GAP)
    ap.add_argument("--rungs", type=int, default=0)
    ap.add_argument("--no-ext", action="store_true",
                    help="현행 60일 박스만 사용 (W-box·A-box 병합 풀 제외)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    use_ext = not args.no_ext
    rungs = args.rungs if args.rungs else max(4, round(COVER / args.gap))

    t0 = tmod.time()
    U = load_universe()
    print(f"[INFO] 유니버스 {len(U):,}종목 ({tmod.time()-t0:.0f}s)", flush=True)
    all_days = np.sort(np.unique(np.concatenate([s["dn"] for s in U])))
    ev = []
    for i, s in enumerate(U):
        for j, knd, sc in zip(s["trig"], s["kinds"], s["sc"]):
            ev.append((int(s["dn"][j]), -int(sc), i, int(j), knd))
    ev.sort()
    ev_days = defaultdict(list)
    for e in ev:
        ev_days[e[0]].append(e)

    if args.sweep:
        rows = []
        for gp in SWEEP_GAPS:
            rg = max(4, round(COVER / gp))
            for mode in ("close", "none", "all"):
                r = run(gp, rg, args.exit, mode, use_ext, U, all_days, ev_days)
                sm = r["summary"]
                rows.append({"gap_pct": gp * 100, "sameday": mode,
                             "rungs": rg, **sm,
                             **{kk: r[kk] for kk in ("avg_capital", "peak_capital",
                                                     "ret_capital_pct",
                                                     "max_rungs")}})
                print(f"  갭 {gp*100:>4.1f}% · {mode:<5} → 총 {sm['net_total']:+,}원 "
                      f"· MDD {sm['mdd_won']:+,} · 자본수익률 {r['ret_capital_pct']:+.1f}%/년",
                      flush=True)
        SWEEP.write_text(json.dumps(
            {"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
             "exit": args.exit, "rows": rows}, ensure_ascii=False),
            encoding="utf-8")
        if not args.no_push:
            subprocess.run(["git", "-C", str(REPO), "add", "-A"], capture_output=True)
            subprocess.run(["git", "-C", str(REPO), "commit", "-m",
                            "chore: 갭 스윕 갱신"], capture_output=True)
            subprocess.run(["git", "-C", str(REPO), "push"], capture_output=True)
        print(f"[INFO] 스윕 완료 → {SWEEP.name}")
        return

    r = run(args.gap, rungs, args.exit, args.sameday, use_ext, U, all_days,
            ev_days, detail=True)
    sens = {}
    for m in ("all", "close", "none"):
        if m == args.sameday:
            continue
        r2 = run(args.gap, rungs, args.exit, m, use_ext, U, all_days, ev_days)
        sens[m] = {"net_total": r2["summary"]["net_total"],
                   "mdd_won": r2["summary"]["mdd_won"],
                   "escape_rate": r2["summary"]["escape_rate"],
                   "turnover_year": r2["summary"]["turnover_year"]}
    s = r["summary"]
    out = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": {"slots": K,
                   "buy_zone": "위치≤50% (W: 주봉밴드 하위 절반 · A: 추세선 근접)"
                               if use_ext else "위치≤50%",
                   "unit": f"{UNIT//10000}만/차",
                   "gap": f"{args.gap*100:g}% 고정",
                   "pool": ("현행60일+W-box(주봉밴드)+A-box(상승저점)" if use_ext
                            else "현행60일"),
                   "start": str(s["start_year"]),
                   "end": str(s["end_year"])},
        "assumptions": {
            "sameday": args.sameday, "sameday_ko": SAMEDAY_KO[args.sameday],
            "exit": args.exit, "exit_ko": EXIT_KO[args.exit],
            "gap_pct": args.gap * 100, "rungs": rungs,
            "unit": UNIT, "fee_roundtrip": FEE,
            "ext_pool": use_ext,
            "ext_ko": ("병합 풀: 현행(60일 Donchian) + W-box(13주 종가 밴드 ±30%·드리프트≤15%·"
                       "밴드 하위 절반 진입·밴드 이탈 청산) + A-box(피벗 저가 3개 연속 상승·"
                       "추세선 근접 진입·선 이탈/붕괴 청산)" if use_ext
                       else "현행 60일 박스만"),
            "entry": "신호일 종가 (이탈일 진입 금지)",
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
          f"평균자본 {r['capital']['avg']:,}원 · 자본수익률 {r['capital']['ret_capital_pct']:+.1f}%/년",
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