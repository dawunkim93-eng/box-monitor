#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""박스권 Top-20(매수존) 순위 진입 시뮬레이션 — 검증 탭 데이터 (box-monitor 전용)
============================================================================
매일: LIVE 박스(주봉 13주 종가 밴드 ±30% · 거래대금 10억+) 중 **매수존(밴드 하위 절반)**
후보 → priority_score 랭킹 → **상위 20 슬롯** 진입. (--pool wa 로 상승저점 포함)
포지션: 그리드 사다리 5차(최심 -9.6%) · 갭 2.5% · next_open 진입 · 종가<경계+2% 또는 소멸 시 청산
· 슬롯 해제 후 다음 날 최고 순위 진입.

■ 현행 가정 (2026-11-06)
   --sameday all|multi|close|none   당일 매도 가정 (기본 multi — 공개 수치가 쓰는 값)
        all  : 지정가 터치 시 체결 (사다리 운용의 표준 모델)
        multi: all + 같은 차수 당일 재왕복 (검증 불가 — 순익의 약 65%가 이 가정에서 발생)
        close: 종가 ≥ 목표가일 때만 당일 매도 (보수)
        none : 당일 신규차수 매도 금지 (하한)
   --near 2.0    경계 N% 위 접근 시 종가 청산 (0=경계 이탈 시만)
   --rungs 0     기본 max(4, round(COVER/gap)) = 5차
   --pool w      주봉박스만 (기본) · wa=주봉+상승저점 · a=상승저점만
   --gap-mode fixed|atr   atr=clip(k×ATR14, 1.5~5%)·near=0.8×갭 (k 0.4~1.2 실측: fixed 우위)
   --entry next_open | close | next_close | limit_signal | limit_line
   --candle all | bear | bear_tail (기본 bear_tail)
   하드스톱 -15%(경계 기준)은 백스톱 — 근접청산(2%)보다 아래라 실측 발동 0건.

■ 산출: 회전 · 탈출율(첫 차수 +갭 매도 비율) · 지속 · 연도별 손익 · 승률 · MDD
   · 청산사유(방향 분리) · 점수구간별 탈출율(score_escape) · 투입자본/자본수익률
   · 가정 명시(assumptions) · 다른 sameday 모드 민감도(sensitivity)

실행: python scripts/box_validation.py [--sweep] [--sameday all] [--out P]
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
GAP = 0.025            # 기본 갭 (gap-mode=fixed)
GAP_MODE = "fixed"     # fixed | atr (atr: gap_i = clip(k × ATR14_i, min, max))
ATR_K = 0.8            # atr 갭 계수
GAP_MIN, GAP_MAX = 0.015, 0.050   # atr 갭 하한/상한
NEAR_OF_GAP = 0.8      # atr 모드 근접청산(%) = 0.8 × 갭 → 1차 체결 전 청산 방지
EXIT_MODE = "intraday"  # close(종가 판정) | intraday(경계 터치 시 장중 청산) — 2026-11 채택
ENTRY_EVAL = True       # 진입일 당일에도 청산 판정 (미적용 시 MDD 과소: -422k vs -649k)
MKT_DN = set()         # 신규 진입 차단일 (지수 급락 쿨다운)
MIN_ROOM = 4.0         # 최소 진입 여지(%) — 진입가에서 경계×1.02까지. 채택값 4% (경계 +6%)
ADAPT_RUNGS = False    # 차수를 room에 맞춰 자동 축소 (floor(room/gap)+1, 최대 5)
AMT_MIN = 1e9
START = 20140901       # 스캔 시작일 (fdr 과거 상한 2014-08)
COVER = 0.125          # 사다리 커버 → rungs = max(4, round(COVER/gap)) = 5차(최심 -9.6%)
                       # 15차(-30%)는 박스 폭(22~30%) 밖: 5차 대비 순익 +6% · MDD +77% (2026-11-06 실측)
POOL = "w"             # 박스 풀: w=주봉박스만(현행) · wa=주봉+상승저점 · a=상승저점만
MAX_TRIPS = 5          # 당일 다중 왕복 상한 (sameday=multi — 같은 차수 재왕복 최대 횟수)
ENTRY_MODE = "close"   # 진입 시점: close(신호일 종가) · next_open(다음날 시가) · next_close(다음날 종가)
ENTRY_ZONE = False     # True면 진입 시점에도 매수존 조건 재확인
CANDLE_MODE = "all"    # all | bear(음봉 신호만) | bear_tail(음봉+아래꼬리1%+)
WBREAK = "close"       # 주봉박스 소멸 판정: close | confirm2(2연속) | buffer1(1% 버퍼)
APOS = "far"           # A박스 위치 점수 방향: far(현행, 선에서 먼 쪽 가점) | near(가까운 쪽 가점)
ROTATE = "keep"        # keep | flat_first | flat_first_excl | flat_first_score
ROT_MARGIN = 2.0       # flat_first_score: 신규 후보 점수 우위 기준
COMP_DUMP = None       # --dump 지정 시 포지션별 진입 지표 기록 (점수 재설계 분석용)
SWEEP_GAPS = [0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]

SAMEDAY_KO = {"all": "낙관(저가→고가 가정)", "close": "현실(종가≥목표가 때만)",
              "none": "보수(당일 신규차수 매도 금지)",
              "multi": "분봉 다중 왕복(당일 범위 기반 재왕복)"}
EXIT_KO = {"close0": "종가 확정 즉시 청산", "close5": "종가 -5% 버퍼",
           "low0": "장중 하단 터치 청산", "low5": "장중 -5% 청산",
           "box": "박스 소멸 시만 (구 규칙)"}


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
        o = df["Open"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        c = df["Close"].values.astype(float)
        bull = c > o
        body = np.where(o > 0, (c - o) / np.maximum(o, 1e-9), 0.0)
        lower_sh = np.where(l > 0,
                            (np.minimum(o, c) - l) / np.maximum(l, 1e-9) * 100,
                            0.0)
        v = df["Volume"].values.astype(float)
        dn = np.array([int(d.strftime("%Y%m%d")) for d in df.index],
                      dtype=np.int64)
        kind_map = {}
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)),
                                          np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = (pd.Series(tr).ewm(alpha=1 / 14, min_periods=14).mean().values
               / np.maximum(c, 1))
        amt20 = pd.Series(c * v).rolling(20).mean().values
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
        wdriftw = np.zeros(nw)      # 13주 순드리프트
        wwidthw = np.zeros(nw)      # 밴드 폭 (최고/최저 - 1)
        for k in range(12, nw):
            seg = wclose[k - 12:k + 1]
            wlo_w[k], whi_w[k] = seg.min(), seg.max()
            wdriftw[k] = seg[-1] / max(seg[0], 1e-9) - 1
            wwidthw[k] = seg.max() / max(seg.min(), 1e-9) - 1
            wflagw[k] = (wwidthw[k] <= 0.30 and abs(wdriftw[k]) <= 0.15)
        # 주봉 기간 — 밴드 조건 연속 유지 주 수
        wrun = np.zeros(nw, dtype=np.int64)
        run_w = 0
        for k in range(nw):
            run_w = run_w + 1 if wflagw[k] else 0
            wrun[k] = run_w
        kk = np.searchsorted(wend, dn) - 1
        okw = kk >= 12
        wdur = np.where(okw, wrun[np.clip(kk, 0, nw - 1)], 0).astype(np.int64)
        wflag = np.where(okw, wflagw[np.clip(kk, 0, nw - 1)], False)
        wlo = np.where(okw, wlo_w[np.clip(kk, 0, nw - 1)], 0.0)
        whi = np.where(okw, whi_w[np.clip(kk, 0, nw - 1)], 0.0)
        wdrift = np.where(okw, wdriftw[np.clip(kk, 0, nw - 1)], 0.0)
        wwidth = np.where(okw, wwidthw[np.clip(kk, 0, nw - 1)], 0.0)
        wpos = np.where(np.maximum(whi - wlo, 1e-9) > 0,
                        (c - wlo) / np.maximum(whi - wlo, 1e-9), 1.0)
        okb = okw & (wwidth <= 0.30) & (np.abs(wdrift) <= 0.15)
        if WBREAK == "close":
            wvalid = wflag
        elif WBREAK == "confirm2":
            out_ = okw & ((c < wlo) | (c > whi))
            prev_out = np.concatenate([[False], out_[:-1]])
            wvalid = okb & ~(out_ & prev_out)
        else:                      # buffer1
            inb = okw & (c >= wlo * 0.99) & (c <= whi * 1.01)
            wvalid = okb & inb
        liq = np.nan_to_num(amt20) >= AMT_MIN
        wdays = np.where(wflag & (wpos <= 0.5) & liq & (dn >= START))[0]
        if POOL in ("wa", "w"):
            for j in wdays:
                kind_map.setdefault(int(j), "wbox")
        # ── A-box (상승 저점 추세선): 피벗 저가 3개 연속 상승 + 종가 ≥ 선×0.995
        #    진입: 추세선 근접(종가 ≤ 선×1.05) · 청산: 하단근접(선+2%) 또는 구조 붕괴
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
        # 상승 기간 — 연속 상승 저점 사슬(정의 B) 경과일
        adur = np.zeros(n, dtype=np.int64)
        if len(PvA) >= 1:
            cstart = np.zeros(len(PvA), dtype=np.int64)
            for k in range(1, len(PvA)):
                cstart[k] = (cstart[k - 1]
                             if PvA[k] > PvA[k - 1] * 1.005 else k)
            kc_all = np.searchsorted(PcA, np.arange(n), side="right") - 1
            okc = kc_all >= 0
            st_idx = PidA[cstart[np.clip(kc_all, 0, len(PvA) - 1)]]
            adur = np.where(okc, np.arange(n) - st_idx, 0).astype(np.int64)
        abox_ok = np.zeros(n, dtype=bool)
        abox_rise = np.zeros(n, dtype=bool)
        tlv = np.full(n, np.nan)
        aslope = np.full(n, np.nan)     # 추세선 기울기 (주당 %)
        adist = np.full(n, np.nan)      # 추세선 대비 거리 (px/tl - 1)
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
            line_ok = okk & rising & (tl_now > 0)      # 구조 유효 = 선이 존재 (경계)
            ok = line_ok & (c >= tl_now * 0.995)       # 라이브/진입 판정에만 가격 테스트
            abox_ok = ok
            abox_rise = okk & rising
            # 경계(tlv)는 종가가 선 아래여도 유지 — 이탈일(가장 필요한 날)에
            # 근접청산·스톱이 NaN으로 꺼지지 않도록 (2026-11-06 수정)
            tlv = np.where(line_ok, tl_now, np.nan)
            aslope = np.where(line_ok, slope / np.maximum(c, 1e-9) * 100 * 5, np.nan)
            adist = np.where(line_ok, c / np.maximum(tl_now, 1e-9) - 1, np.nan)
        # 상승 스윙 높이 — 사슬 시작 저점 대비 현재 고가 (구조 고점 배율)
        aheight = np.zeros(n)
        if len(PvA) >= 1:
            for seg_start in np.unique(st_idx[okc]):
                idx = np.where(okc & (st_idx == seg_start))[0]
                if not len(idx):
                    continue
                a, b = int(idx[0]), int(idx[-1]) + 1
                base = max(float(l[seg_start]), 1e-9)
                aheight[a:b] = np.maximum.accumulate(h[a:b]) / base - 1
        atrig = (abox_ok & (c <= np.nan_to_num(tlv, nan=0) * 1.05)
                 & (np.nan_to_num(tlv, nan=0) > 0)
                 & liq & (dn >= START))
        for j in (np.where(atrig)[0] if POOL in ("wa", "a") else []):
            kind_map.setdefault(int(j), "abox")
        ti = np.array(sorted(kind_map), dtype=np.int64)
        kinds_arr = [kind_map[int(j)] for j in ti]
        if not len(ti):
            continue
        def _comp(j, knd):
            if knd == "wbox":
                return (float(wpos[j]),
                        float(atr[j]) if not np.isnan(atr[j]) else None,
                        float(wdrift[j]), float(wwidth[j]))
            if knd == "abox":
                return (float(adist[j]) if not np.isnan(adist[j]) else None,
                        float(atr[j]) if not np.isnan(atr[j]) else None,
                        float(aslope[j]) if not np.isnan(aslope[j]) else None,
                        float(aheight[j]))
            return (None, None, None, None)
        sc = np.array([score_new(knd, *_comp(j, knd))
                       for j, knd in zip(ti, kinds_arr)], dtype=np.int64)
        U.append({"code": code, "name": name, "dn": dn, "o": o, "h": h, "l": l,
                  "c": c, "atr": atr, "amt20": amt20,
                  "bull": bull, "body": body, "lower_sh": lower_sh,
                  "wflag": wflag, "wvalid": wvalid, "wlo": wlo, "whi": whi, "wpos": wpos,
                  "wdur": wdur, "adur": adur, "wdrift": wdrift, "wwidth": wwidth,
                  "adist": adist, "aslope": aslope, "aheight": aheight,
                  "abox_ok": abox_ok, "abox_rise": abox_rise, "tlv": tlv,
                  "trig": ti, "kinds": kinds_arr, "sc": sc})
    return U


SCHEME = "S2b"  # 점수안: S2b 추세우선+W감점 (최종) · S1 균형 · S2 추세 · S3 ATR우선


def score_new(kind, pos, atr, trend, height):
    """W/A 점수 개편안 (2026-11-02 13년 실측 버킷 기준 · 만점 6).
    W: 위치(<20% 0 · ~35% 1 · ~50% 2) / 추세(<0 0 · ~5% 1 · 5%+ 2)
       / ATR(4~7% 1, 역U) / 높이(밴드폭 22%+ 1)
    A: 거리(≤2% 0 · ~3.5% 1 · ~5% 2) / ATR(<5% 0 · ~6% 1 · 6%+ 2)
       / 기울기(<2.2%/주 0 · ~3 1 · 3%+ 2)
    """
    if kind == "wbox":
        p = 0 if pos is None or pos < 0.20 else (1 if pos < 0.35 else 2)
        t = 0 if trend is None or trend < 0 else (1 if trend < 0.05 else 2)
        if SCHEME in ("S2", "S2b"):
            a = 1 if (atr is not None and 0.04 <= atr < 0.07) else 0
            h = 1 if (height is not None and height >= 0.22) else 0
            s = p + t + a + h
            if SCHEME == "S2b" and trend is not None and trend < -0.05:
                s -= 1
            return s
        if SCHEME == "S3":       # ATR우선
            a = 0 if atr is None or atr < 0.04 else (1 if atr < 0.05 else (2 if atr < 0.07 else 3))
            t = 0 if trend is None or trend < 0.05 else 1
            return p + a + t
        # S1 균형
        a = 0 if atr is None or atr < 0.04 else (2 if atr < 0.07 else 1)
        t = 1 if (trend is not None and trend >= 0.05) else 0
        h = 1 if (height is not None and height >= 0.22) else 0
        return p + a + t + h
    if kind == "abox":
        if APOS == "near":
            p = (2 if (pos is not None and pos <= 0.02)
                 else (1 if (pos is not None and pos < 0.035) else 0))
        else:
            p = 0 if pos is None or pos <= 0.02 else (1 if pos < 0.035 else 2)
        if SCHEME == "S3":
            a = 0 if atr is None or atr < 0.05 else (1 if atr < 0.06 else (2 if atr < 0.07 else 3))
            t = 0 if trend is None or trend < 2.2 else 1
            return p + a + t
        a = 0 if atr is None or atr < 0.05 else (1 if atr < 0.06 else 2)
        if SCHEME in ("S2", "S2b"):
            t = 0 if trend is None or trend < 2.2 else (1 if trend < 3.0 else 2)
            return p + t + a
        # S1 균형
        t = 1 if (trend is not None and trend >= 2.2) else 0
        return p + a + t
    return 0


def run(gap, rungs, near, sameday, U, all_days, ev_days,
        detail=False):
    """한 변형 실행. near=하단 경계 N% 위 접근 시 조기 청산(%).
    상세(detail=True)면 사이트 렌더용 전체 블록 반환."""
    holdings = {}
    done, curve = [], []
    cap_daily, max_pos, max_rungs = [], 0, 0
    K_ = K

    def close_position(done, holdings, curve, i, px, why, day):
        hst = holdings.pop(i)
        for k in range(len(hst["held"])):
            if hst["held"][k]:
                hst["net"] += (px - hst["levels"][k]) / hst["levels"][k] * UNIT
                if k == 0 and hst["grid_n"] == 1:
                    hst["loss0"] = True
        done.append({"code": i, "entry_day": hst["entry_day"],
                     "exit_day": int(day), "score": hst["score"],
                     "escape0": hst["escape0"], "loss0": hst["loss0"],
                     "cycles": hst["cycles"], "net": hst["net"],
                     "days": hst["j"] - hst["j_entry"], "kind": why,
                     "ekind": hst.get("kind", "std"),
                     "dur": hst.get("dur", 0),
                     "cbull": hst.get("cbull"),
                     "cbody": hst.get("cbody"),
                     "clower": hst.get("clower")})
        if COMP_DUMP is not None:
            COMP_DUMP.append({
                "code": i, "exit_day": int(day),
                "entry_day": hst["entry_day"], "kind": hst.get("kind", "std"),
                "pos": hst.get("cpos"), "atr": hst.get("catr"),
                "trend": hst.get("ctrend"), "height": hst.get("cheight"),
                "score": hst["score"], "escape0": hst["escape0"],
                "loss0": hst["loss0"], "net": round(hst["net"]),
                "days": hst["j"] - hst["j_entry"], "cycles": hst["cycles"],
                "exit": why, "cbull": hst.get("cbull"),
                "cbody": hst.get("cbody"), "clower": hst.get("clower")})
        curve.append((int(day), hst["net"]))

    for D in all_days:
        if D < START:
            continue
        rotated_today = set()
        for i in list(holdings):
            hst, s = holdings[i], U[i]
            dn = s["dn"]
            j = int(np.searchsorted(dn, D))
            if j >= len(dn) or dn[j] != D:
                close_position(done, holdings, curve, i,
                               float(s["c"][hst["j"]]), "데이터갭", D)
                continue
            if j < hst["j_entry"]:
                continue
            if j == hst["j_entry"] and ENTRY_MODE == "next_close":
                hst["j"] = j         # 종가 진입일 — 당일 체결 없음
                continue
            knd = hst.get("kind")
            wlo = float(s["wlo"][j])
            whi = float(s["whi"][j])
            tlv = (float(s["tlv"][j]) if not np.isnan(s["tlv"][j]) else 0.0)
            px, lw = float(s["c"][j]), float(s["l"][j])
            g_i, n_i = hst.get("gap", gap), hst.get("near", near)
            # 1) 하단 near% 위 근접 청산 — wbox=밴드 하단 · abox=추세선
            #    near는 포지션별: 고정모드=near · ATR모드=NEAR_OF_GAP×갭
            B_ = (wlo if knd == "wbox" else tlv)
            B_ = B_ if B_ > 0 else None
            if B_ is not None:
                lvl = B_ * (1 + n_i / 100)
                if EXIT_MODE == "intraday":
                    o_j = float(s["o"][j])
                    fill = (o_j if (o_j > 0 and o_j < lvl)
                            else (lvl if lw < lvl else None))
                    if fill is not None:
                        close_position(done, holdings, curve, i, fill,
                                       "하단근접(장중)", D)
                        continue
                elif px < lvl:
                    close_position(done, holdings, curve, i, px, "하단근접 청산", D)
                    continue
            # 2) 소멸 — wbox: 밴드 유지·상방 / abox: 구조 붕괴
            if knd == "wbox":
                if WBREAK == "close":
                    gone = (not bool(s["wflag"][j])) or (whi > 0 and px > whi)
                else:
                    gone = not bool(s["wvalid"][j])
                if gone:
                    close_position(done, holdings, curve, i, px, "주봉박스 소멸", D)
                    continue
            if knd == "abox" and not bool(s["abox_rise"][j]):
                close_position(done, holdings, curve, i, px, "추세선 붕괴", D)
                continue
            hst["j"] = j
            # 3) 하드스톱 백스톱 (-15%) — 타입 경계 기준
            stop_px = (B_ * 0.85) if B_ is not None else None
            if stop_px is not None and lw <= stop_px:
                close_position(done, holdings, curve, i, stop_px, "손절-15%", D)
                continue
            # 4) 그리드 진행 — 저가 채결(매수) → 고가 채결(매도)
            levels, held = hst["levels"], hst["held"]
            bought = set()
            for k in range(1, len(held)):
                if not held[k] and lw <= levels[k]:
                    held[k] = True
                    bought.add(k)
            for k in range(len(held)):
                if not held[k]:
                    continue
                tgt = levels[k] * (1 + g_i)
                if float(s["h"][j]) < tgt:
                    continue
                if (k in bought) and sameday not in ("all", "multi"):
                    if sameday == "none":
                        continue
                    if sameday == "close" and px < tgt:
                        continue
                hst["net"] += g_i * UNIT - FEE * UNIT
                held[k] = False
                hst["cycles"] += 1
                if k == 0 and hst["grid_n"] == 1:
                    hst["escape0"] = True
                # 같은 날 같은 차수 재왕복 — 당일 범위에서 2×갭 이동당 1회 추가
                # (저가 매수→고가 매도→재매수→재매도 경로 가정 · 분봉 다중 체결 근사)
                if sameday == "multi":
                    rng_d = float(s["h"][j]) - float(s["l"][j])
                    extra = int(rng_d // (2 * levels[k] * g_i))
                    extra = min(max(extra, 0), MAX_TRIPS)
                    if extra:
                        hst["net"] += extra * (g_i * UNIT - FEE * UNIT)
                        hst["cycles"] += extra
            if not any(held):      # 재눌림 → 재그리드/회전 (타입별 매수존 복귀)
                if ROTATE in ("flat_first", "flat_first_excl") and hst.get("escape0"):
                    close_position(done, holdings, curve, i, px, "1차탈출 회전", D)
                    rotated_today.add(i)
                    continue
                if ROTATE == "flat_first_score" and hst.get("escape0"):
                    base_sc = hst.get("score") or 0
                    better = any(
                        (-nsc2) >= base_sc + ROT_MARGIN
                        and i2 not in holdings and i2 != i
                        and i2 not in rotated_today
                        for d2, nsc2, i2, j2, knd2 in ev_days.get(int(D), []))
                    if better:
                        close_position(done, holdings, curve, i, px, "1차탈출 회전", D)
                        rotated_today.add(i)
                        continue
                if knd == "wbox":
                    in_zone = (float(s["wpos"][j]) <= 0.5)
                else:
                    ad_ = float(s["adist"][j])
                    in_zone = (not np.isnan(ad_)) and ad_ <= 0.05
                if in_zone:
                    hst["grid_n"] += 1
                    nr = len(hst["held"])
                    if ADAPT_RUNGS and knd == "wbox":
                        _b = float(s["wlo"][j])
                        if _b > 0:
                            _rm = (px / (_b * 1.02) - 1) * 100
                            nr = int(min(max(int(_rm // (g_i * 100)) + 1, 1), rungs))
                    hst["levels"] = [px * (1 - g_i) ** k for k in range(nr)]
                    hst["held"] = [False] * nr
                    hst["held"][0] = True
            held_cnt = sum(holdings[i]["held"])
            max_rungs = max(max_rungs, held_cnt)
        cap_daily.append(sum(
            sum(h["held"]) for h in holdings.values()) * UNIT)
        max_pos = max(max_pos, len(holdings))
        free = K_ - len(holdings)
        if free > 0 and not (MKT_DN and int(D) in MKT_DN):
            for d, nsc, i, j, knd in ev_days.get(int(D), []):
                if free <= 0:
                    break
                if i in holdings:
                    continue
                if ROTATE in ("flat_first_excl", "flat_first_score") and i in rotated_today:
                    continue
                s = U[i]
                dn_s = s["dn"]
                if ENTRY_MODE in ("limit_signal", "limit_line"):
                    je = j + 1
                    if je >= len(dn_s):
                        continue
                    if ENTRY_MODE == "limit_signal":
                        lim = float(s["c"][j])          # 신호일 종가 지정가
                    else:
                        lim = (float(s["wlo"][je]) * 1.03
                               if knd == "wbox" else float(s["tlv"][je]) * 1.03)
                    if not (lim > 0):
                        continue
                    if float(s["l"][je]) > lim:         # 미체결 → 스킵
                        continue
                    px_e = min(float(s["o"][je]), lim)  # 갭다운 시 시가 체결
                else:
                    je = j if ENTRY_MODE == "close" else j + 1
                    if je >= len(dn_s):
                        continue
                    px_e = (float(s["c"][je])
                            if ENTRY_MODE in ("close", "next_close")
                            else float(s["o"][je]))
                if not (px_e > 0):
                    continue
                if MIN_ROOM > 0 and knd == "wbox":
                    _w0 = float(s["wlo"][je])
                    if _w0 > 0 and (px_e / (_w0 * 1.02) - 1) * 100 < MIN_ROOM:
                        continue
                # 포지션별 갭 — ATR 모드: 신호일 ATR 기준(진입일 종가 미사용 = 룩어헤드 없음)
                if GAP_MODE == "atr":
                    a_sig = float(s["atr"][j])
                    gi = (min(max(ATR_K * a_sig, GAP_MIN), GAP_MAX)
                          if not np.isnan(a_sig) else gap)
                    n_i = NEAR_OF_GAP * gi * 100
                else:
                    gi, n_i = gap, near
                if CANDLE_MODE != "all":        # 신호일 캔들 필터
                    if not (float(s["c"][j]) < float(s["o"][j])):
                        continue
                    if CANDLE_MODE == "bear_tail" \
                            and float(s["lower_sh"][j]) < 1.0:
                        continue
                if ENTRY_ZONE:          # 진입 시점 매수존 재확인 (추격 금지)
                    if knd == "wbox":
                        w0, w1 = float(s["wlo"][je]), float(s["whi"][je])
                        if not (w1 > w0 > 0):
                            continue
                        if not ((px_e - w0) / (w1 - w0) <= 0.5
                                and px_e >= w0 * (1 + n_i / 100)):
                            continue
                    else:
                        t0 = float(s["tlv"][je])
                        if not (t0 > 0):
                            continue
                        dz = px_e / t0 - 1
                        if not (n_i / 100 <= dz <= 0.05):
                            continue
                ng = rungs
                if ADAPT_RUNGS and knd == "wbox":
                    _w0b = float(s["wlo"][je])
                    if _w0b > 0:
                        _rm = (px_e / (_w0b * 1.02) - 1) * 100
                        ng = int(min(max(int(_rm // (gi * 100)) + 1, 1), rungs))
                holdings[i] = {"j": je, "j_entry": je,
                               "entry_day": int(dn_s[je]),
                               "entry": px_e, "score": -nsc,
                               "gap": gi, "near": n_i, "kind": knd,
                               "dur": (int(s["wdur"][je]) if knd == "wbox"
                                       else int(s["adur"][je]) if knd == "abox"
                                       else 0),
                               "cpos": (float(s["wpos"][je]) if knd == "wbox"
                                        else (float(s["adist"][je])
                                              if knd == "abox"
                                              and not np.isnan(s["adist"][je])
                                              else None)),
                               "catr": (float(s["atr"][je])
                                        if not np.isnan(s["atr"][je]) else None),
                               "ctrend": (float(s["wdrift"][je]) if knd == "wbox"
                                          else (float(s["aslope"][je])
                                                if knd == "abox"
                                                and not np.isnan(s["aslope"][je])
                                                else None)),
                               "cheight": (float(s["wwidth"][je])
                                           if knd == "wbox"
                                           else (float(s["aheight"][je])
                                                 if knd == "abox" else None)),
                               "cbull": bool(s["bull"][j]),
                               "cbody": float(s["body"][j]),
                               "clower": float(s["lower_sh"][j]),
                               "levels": [px_e * (1 - gi) ** k
                                          for k in range(ng)],
                               "held": [False] * ng,
                               "grid_n": 1, "escape0": False, "loss0": False,
                               "cycles": 0, "net": 0.0}
                holdings[i]["held"][0] = True
                free -= 1
                if ENTRY_EVAL:      # 진입일 당일 청산 판정 (매수 직후 밀리는 경우)
                    b_e = (float(s["wlo"][je]) if knd == "wbox"
                           else float(s["tlv"][je]))
                    if b_e > 0:
                        lvl_e = b_e * (1 + n_i / 100)
                        o_e, l_e, c_e = (float(s["o"][je]), float(s["l"][je]),
                                         float(s["c"][je]))
                        if EXIT_MODE == "intraday":
                            fill = (o_e if (o_e > 0 and o_e < lvl_e)
                                    else (lvl_e if l_e < lvl_e else None))
                        else:
                            fill = c_e if c_e < lvl_e else None
                        if fill is not None:
                            close_position(done, holdings, curve, i, fill,
                                           "하단근접(당일)", D)
                            free += 1
                            continue
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
                    "ret_peak_pct": round(summary["net_total"]
                                          / max(pk_cap, 1) / years * 100, 1),
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

    def dstat(ps):
        return {"n": len(ps),
                "esc_rate": round(sum(1 for p in ps if p["escape0"]) / len(ps) * 100, 1) if ps else None,
                "avg_net": round(sum(p["net"] for p in ps) / len(ps)) if ps else None,
                "loss0_rate": round(sum(1 for p in ps if p["loss0"]) / len(ps) * 100, 1) if ps else None}

    dur_escape = {}
    for kd, edges, labels in (
            ("wbox", [-1, 4, 8, 16, 26, 10 ** 9],
             ["≤4주", "5~8주", "9~16주", "17~26주", "27주+"]),
            ("abox", [-1, 10, 30, 60, 120, 10 ** 9],
             ["≤10일", "11~30일", "31~60일", "61~120일", "121일+"])):
        ps = [p for p in done if p.get("ekind") == kd]
        dur_escape[kd] = {
            labels[b]: dstat([p for p in ps if edges[b] < p["dur"] <= edges[b + 1]])
            for b in range(len(labels))}

    candle_escape = {}
    for kd in ("wbox", "abox", "all"):
        ps = done if kd == "all" else [p for p in done if p.get("ekind") == kd]
        blk = {}
        for lab, flag in (("양봉", True), ("음봉", False)):
            blk[lab] = dstat([p for p in ps if p.get("cbull") is flag])
        nb = [p for p in ps if p.get("cbull") is False]
        blk["음봉·아래꼬리1%+"] = dstat([p for p in nb
                                        if (p.get("clower") or 0) >= 1.0])
        blk["음봉·아래꼬리<1%"] = dstat([p for p in nb
                                       if (p.get("clower") or 0) < 1.0])
        candle_escape[kd] = blk

    candle_yearly = {}
    for yr in sorted({p["entry_day"] // 10000 for p in done}):
        ps = [p for p in done if p["entry_day"] // 10000 == yr]
        candle_yearly[str(yr)] = {
            lab: dstat([p for p in ps if p.get("cbull") is flag])
            for lab, flag in (("양봉", True), ("음봉", False))}

    dur_score_x = {}
    for kd in ("wbox", "abox"):
        ps = [p for p in done if 5 <= p["score"] <= 6 and p.get("ekind") == kd]
        if not ps:
            continue
        ds = sorted(p["dur"] for p in ps)
        med = ds[len(ds) // 2]
        dur_score_x[kd] = {"median": med,
                           "short": dstat([p for p in ps if p["dur"] <= med]),
                           "long": dstat([p for p in ps if p["dur"] > med])}
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
        "dur_escape": dur_escape,
        "dur_score_x": dur_score_x,
        "candle_escape": candle_escape,
        "candle_yearly": candle_yearly,
        "capital": {"avg": round(av_cap), "peak": round(pk_cap),
                    "p95": round(p95_cap),
                    "ret_capital_pct": round(summary["net_total"]
                                             / max(av_cap, 1) / years * 100, 1),
                    "ret_peak_pct": round(summary["net_total"]
                                          / max(pk_cap, 1) / years * 100, 1),
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
    global MAX_TRIPS, COMP_DUMP, SCHEME, ENTRY_MODE, ENTRY_ZONE, CANDLE_MODE, WBREAK, APOS, ROTATE, ROT_MARGIN, POOL
    global GAP_MODE, ATR_K, NEAR_OF_GAP, GAP_MIN, GAP_MAX
    global EXIT_MODE, ENTRY_EVAL, MKT_DN
    global MIN_ROOM
    global ADAPT_RUNGS
    ap = argparse.ArgumentParser(description="박스 Top-20 검증 시뮬레이션 (W/A 풀 · 하단근접 청산)")
    ap.add_argument("--sameday", choices=["close", "none", "all", "multi"],
                    default="multi",
                    help="당일 체결 가정 (기본 multi=분봉 다중 왕복 근사)")
    ap.add_argument("--gap", type=float, default=GAP)
    ap.add_argument("--gap-mode", choices=["fixed", "atr"], default=GAP_MODE,
                    help="갭 설정 (fixed=고정 · atr=k×ATR14 종목별)")
    ap.add_argument("--atr-k", type=float, default=ATR_K,
                    help="atr 갭 계수 (기본 0.8)")
    ap.add_argument("--near-of-gap", type=float, default=NEAR_OF_GAP,
                    help="atr 모드 근접청산(%%) = near-of-gap × 갭 (기본 0.8)")
    ap.add_argument("--gap-min", type=float, default=GAP_MIN)
    ap.add_argument("--gap-max", type=float, default=GAP_MAX)
    ap.add_argument("--rungs", type=int, default=0)
    ap.add_argument("--near", type=float, default=2.0,
                    help="하단 경계 N%% 위 접근 시 조기 청산 (%%) · 0=이탈 완료 시만")
    ap.add_argument("--exit-mode", choices=["close", "intraday"], default=EXIT_MODE,
                    help="청산 판정 (close=종가 · intraday=경계 터치 시 장중)")
    ap.add_argument("--entry-eval", action=argparse.BooleanOptionalAction,
                    default=ENTRY_EVAL, help="진입일 당일 청산 판정 포함")
    ap.add_argument("--mkt-csv", type=str, default="",
                    help="지수 일봉 CSV (Date,Close) — 급락 쿨다운용")
    ap.add_argument("--cool", type=int, default=0,
                    help="지수 -cool-thr%% 이하 마감 후 신규 진입 중단 일수")
    ap.add_argument("--cool-thr", type=float, default=3.0)
    ap.add_argument("--min-room", type=float, default=MIN_ROOM,
                    help="최소 진입 여지(%%) — 진입가→경계×1.02 거리")
    ap.add_argument("--adapt-rungs", action=argparse.BooleanOptionalAction,
                    default=ADAPT_RUNGS,
                    help="차수를 room에 맞춰 축소 (floor(room/gap)+1, 최대 5)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--scheme", choices=["S1", "S2", "S2b", "S3"],
                    default="S2b", help="점수안 (S2b 최종)")
    ap.add_argument("--entry",
                    choices=["close", "next_open", "next_close",
                             "limit_signal", "limit_line"],
                    default="next_open",
                    help="진입 시점 (close/next_open/next_close · "
                         "limit_signal=신호종가 지정가 · limit_line=경계+3% 지정가)")
    ap.add_argument("--apos", choices=["far", "near"], default="far",
                    help="A박스 위치 점수 방향 (far=먼 쪽 가점(현행) · near=가까운 쪽 가점)")
    ap.add_argument("--rotate",
                    choices=["keep", "flat_first", "flat_first_excl", "flat_first_score"],
                    default="keep",
                    help="1차 탈출 후 슬롯 처리 (keep=재그리드 · flat_first=플랫 시 회전 · "
                         "flat_first_excl=회전 당일 동일종목 배제 · flat_first_score=점수 우위 시만)")
    ap.add_argument("--rotate-margin", type=float, default=2.0,
                    help="flat_first_score 점수 우위 기준 (기본 2점)")
    ap.add_argument("--wbreak", choices=["close", "confirm2", "buffer1"],
                    default="close",
                    help="주봉박스 소멸 판정 (close=일봉 즉시 · confirm2=2연속 · buffer1=1% 버퍼)")
    ap.add_argument("--candle", choices=["all", "bear", "bear_tail"],
                    default="bear_tail",
                    help="신호일 캔들 필터 (bear=음봉만 · bear_tail=음봉+아래꼬리1%+)")
    ap.add_argument("--entry-zone", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="진입 시점에도 매수존 조건 재확인 (기본 on · --no-entry-zone으로 해제)")
    ap.add_argument("--dump", type=str, default="",
                    help="포지션별 진입 지표 JSON 저장 (점수 재설계 분석용)")
    ap.add_argument("--max-trips", type=int, default=1,
                    help="sameday=multi 시 차수당 당일 추가 왕복 상한 (기본 1)")
    ap.add_argument("--pool", choices=["w", "wa", "a"], default=POOL,
                    help="박스 풀 (w=주봉박스만(기본) · wa=주봉+상승저점 · a=상승저점만)")
    args = ap.parse_args()
    POOL = args.pool
    GAP_MODE = args.gap_mode
    ATR_K = args.atr_k
    NEAR_OF_GAP = args.near_of_gap
    GAP_MIN, GAP_MAX = args.gap_min, args.gap_max
    EXIT_MODE = args.exit_mode
    ENTRY_EVAL = args.entry_eval
    MIN_ROOM = args.min_room
    ADAPT_RUNGS = args.adapt_rungs
    if args.mkt_csv and args.cool > 0:
        _m = pd.read_csv(args.mkt_csv, parse_dates=["Date"]).set_index("Date")
        _dn = [int(d.strftime("%Y%m%d")) for d in _m.index]
        _ret = _m["Close"].pct_change() * 100
        _blk = set()
        for _i in range(1, len(_dn)):
            if _ret.iloc[_i] <= -args.cool_thr:
                for _k in range(1, args.cool + 1):
                    if _i + _k < len(_dn):
                        _blk.add(_dn[_i + _k])
        MKT_DN = _blk
        print(f"[INFO] 급락 쿨다운 차단일 {len(_blk)}일 (thr -{args.cool_thr:g}%)")
    SCHEME = args.scheme
    ENTRY_MODE = args.entry
    ENTRY_ZONE = args.entry_zone
    CANDLE_MODE = args.candle
    WBREAK = args.wbreak
    APOS = args.apos
    ROTATE = args.rotate
    ROT_MARGIN = args.rotate_margin
    MAX_TRIPS = max(0, args.max_trips)
    COMP_DUMP = [] if args.dump else None
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
                r = run(gp, rg, args.near, mode,
                        U, all_days, ev_days)
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

    r = run(args.gap, rungs, args.near, args.sameday, U, all_days,
            ev_days, detail=True)
    dump_main, COMP_DUMP_prev = COMP_DUMP, COMP_DUMP
    COMP_DUMP = None      # 민감도 재실행분은 덤프 제외
    sens = {}
    for m in ("all", "close", "none"):
        if m == args.sameday:
            continue
        r2 = run(args.gap, rungs, args.near, m, U, all_days, ev_days)
        sens[m] = {"net_total": r2["summary"]["net_total"],
                   "mdd_won": r2["summary"]["mdd_won"],
                   "escape_rate": r2["summary"]["escape_rate"],
                   "turnover_year": r2["summary"]["turnover_year"]}
    s = r["summary"]
    out = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": {"slots": K,
                   "buy_zone": ("주봉: 밴드 하위 절반" if POOL == "w" else
                                "주봉: 밴드 하위 절반 · 상승: 추세선 근접 ≤5%"),
                   "unit": f"{UNIT//10000}만/차",
                   "gap": f"{args.gap*100:g}% 고정",
                   "rotate": ROTATE,
                   "rotate_margin": ROT_MARGIN,
                   "pool": {"w": "주봉박스(13주 종가밴드)만",
                            "wa": "주봉박스+상승저점",
                            "a": "상승저점만"}[POOL],
                   "start": str(s["start_year"]),
                   "end": str(s["end_year"])},
        "assumptions": {
            "sameday": args.sameday, "sameday_ko": SAMEDAY_KO[args.sameday],
            "near": args.near,
            "near_ko": (f"청산 = 전일 종가가 각 타입 경계(밴드하단/추세선)의 "
                        f"{args.near:g}% 위에 접근하면 조기 청산" if args.near > 0
                        else "청산 = 전일 종가가 경계 하향 이탈(버퍼 없음)"),
            "gap_pct": args.gap * 100, "rungs": rungs,
            "gap_mode": GAP_MODE,
            "exit_mode": EXIT_MODE, "entry_eval": ENTRY_EVAL,
            "cool": args.cool, "cool_thr": args.cool_thr,
            "atr_k": (ATR_K if GAP_MODE == "atr" else None),
            "near_of_gap": (NEAR_OF_GAP if GAP_MODE == "atr" else None),
            "unit": UNIT, "fee_roundtrip": FEE,
            "pool_ko": ("주봉박스(13주 종가 밴드 ±30%·드리프트≤15%·밴드 하위 절반 진입·"
                        "밴드 이탈 소멸) + 상승저점(피벗 저가 3개 연속 상승·추세선 근접 "
                        "진입·선 이탈/붕괴 소멸)"),
            "entry": f"{args.entry} (진입 시점; 신호일 캔들 필터={args.candle})",
            "stop": "전일 박스 하단 -15% 백스톱",
            "escape_def": "탈출율 = 첫 차수(진입)가 +갭 가격에 매도된 비율"},
        "sensitivity": {m: {"sameday_ko": SAMEDAY_KO[m], **v} for m, v in sens.items()},
        **{kk: r[kk] for kk in ("summary", "yearly", "exit_kinds", "score_escape",
                                "dur_escape", "dur_score_x", "candle_escape",
                                "candle_yearly",
                                "capital", "recent_positions", "stock_history")},
    }
    print(f"[INFO] 포지션 {s['positions']:,} ({s['start_year']}~{s['end_year']}"
          f" {s['years']}년) · 탈출율 {s['escape_rate']}% · 지속 {s['hold_avg']}일 · "
          f"총 {s['net_total']:+,}원 · MDD {s['mdd_won']:+,}원"
          f"({s['mdd_vs_capital_pct']}%) · 적자연도 {s['loss_years']}/{s['n_years']} · "
          f"평균자본 {r['capital']['avg']:,}원 · peak {r['capital']['peak']:,}원"
          f" · 자본수익률(peak) {r['capital']['ret_peak_pct']:+.1f}%/년",
          flush=True)
    print("[기간 버킷]", flush=True)
    for kd, blk in r.get("dur_escape", {}).items():
        print(f"  [{kd}]", flush=True)
        for lab, st in blk.items():
            print(f"    {lab:<8} n={st['n']:>5} · 탈출 {st['esc_rate']}% · "
                  f"건당 {st['avg_net']}원 · 손실0 {st['loss0_rate']}%", flush=True)
    print("[전일 캔들]", flush=True)
    for kd, blk in r.get("candle_escape", {}).items():
        print(f"  [{kd}]", flush=True)
        for lab, st in blk.items():
            print(f"    {lab:<14} n={st['n']:>5} · 탈출 {st['esc_rate']}% · "
                  f"건당 {st['avg_net']}원 · 손실0 {st['loss0_rate']}%", flush=True)
    print("[전일 캔들 연도별]", flush=True)
    for yy, row in r.get("candle_yearly", {}).items():
        b, e = row["양봉"], row["음봉"]
        print(f"  {yy}: 양봉 n={b['n']:>4} 탈출{b['esc_rate']}%/{b['avg_net']}원 · "
              f"음봉 n={e['n']:>4} 탈출{e['esc_rate']}%/{e['avg_net']}원", flush=True)
    print("[5~6점 구간 내 기간 효과]", flush=True)
    for kd, st in r.get("dur_score_x", {}).items():
        print(f"  [{kd}] 중앙 {st['median']} · 짧음 {st['short']} · 김 {st['long']}",
              flush=True)
    if args.dump and dump_main:
        Path(args.dump).write_text(json.dumps(dump_main, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"[INFO] 지표 덤프 {args.dump} ({len(dump_main):,}건)", flush=True)
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