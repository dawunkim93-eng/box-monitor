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
import os
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
K = 7                  # 동시 보유 슬롯 (무료 매직스플릿 7슬롯)
GAP = 0.015            # 기본 갭 (gap-mode=fixed) — 2026-11-13 1.5% 채택 (탈출률 79%, 순익 +20%)
GAP_MODE = "fixed"     # fixed | atr (atr: gap_i = clip(k × ATR14_i, min, max))
ATR_K = 0.8            # atr 갭 계수
GAP_MIN, GAP_MAX = 0.015, 0.050   # atr 갭 하한/상한
NEAR_OF_GAP = 0.8      # atr 모드 근접청산(%) = 0.8 × 갭 → 1차 체결 전 청산 방지
EXIT_MODE = "close"     # close(종가 판정, 2026-11-10 채택) | intraday(경계 터치 시 장중 청산)
ENTRY_EVAL = True       # 진입일 당일에도 청산 판정 (미적용 시 MDD 과소: -422k vs -649k)
MKT_DN = set()         # 신규 진입 차단일 (지수 급락 쿨다운)
POS_MAX = 50.0         # 진입 시 밴드 위치 상한(%) — 낮을수록 박스 하단 근처에서만
BOX_DUR = 0            # 박스 최소 유지 주 수 (0=제한 없음)
TIME_STOP = 0          # 보유 최대 거래일 (0=제한 없음)
VOL_MAX = 0.0          # 신호일 거래량 / 20일 평균 상한 (0=제한 없음)
HARD_STOP = 15.0       # 경계 대비 하드스톱 (%)
SIZE_MULT_OVR = []     # --size-mult 지정 시 차수별 배수 덮어쓰기
RUNG_FILL = [0] * 12   # 차수별 체결 횟수 (진단용)
MIN_ROOM = 6.0         # 최소 진입 여지(%) — 진입가에서 청산선까지. 채택값 6% (경계 +8%, 2026-11-13)
LO52_MIN = 0.0         # 52주 저가 대비 최소 거리(%) — 이보다 가까우면 제외
RANK = "score"         # 같은 날 후보 정렬: score(높은 점수 먼저) | pos(밴드 하단에 가까운 순)
EXIT_BUF = 2.0         # 청산선 = 밴드하단×(1+EXIT_BUF/100). +2=하단 2% 위 · -2=하단 2% 아래
EXIT_AT = "close"      # 청산 체결: close(당일 종가) | next_open(다음날 시가)
PB_MODE = False        # 눌림+박스 유지 진입 허용 (20일선 지지 · 60일선 위 상승)
PB_EXIT_LINE = "band"  # 눌림 포지션 청산선: band(밴드 하단) | ma20(진입 시점 20일선)
PB_SLOTS = 0           # 눌림 전용 슬롯 수 (0=구분 없음 · 주봉은 K_-PB_SLOTS 사용)
RV_MODE = False        # 반전 돌파 진입 (장기 하락 + 바닥 + 거래량 폭발 + 하락 추세 돌파)
RV_SLOTS = 0           # 반전 전용 슬롯 수 (0=구분 없음)
AB_PIN = False         # A박스 추세선을 진입 시점 값으로 고정
VP_MODE = "off"        # 매물대 필터: off | below(하단<매물대중심) | above | near(±3%)
VP_BOOST = 2.0         # 박스 하단이 매물대 위일 때 슬롯 금액 배수 (2026-11-12 채택)
VP_WIN = 250           # 매물대(VWAP) 기간(거래일)
BOX_SRC = "band"      # 박스 정의: band(주봉 6주) | va(1년 거래대금 90% 혼잡 구간)
VA_PCT = 0.90          # 혼잡 구간 비율
VP_SCORE = False       # 매물대 거리로 점수 가감 (+2/+1/0/-1)
VP_TIER = False        # 매물대 거리로 금액 차등 (×1/×1.5/×2/×0.8)
ADAPT_RUNGS = False    # 차수를 room에 맞춰 자동 축소 (floor(room/gap)+1, 최대 5)
MIN_SCORE = 0          # 최소 진입 점수 (0=비활성)
SLOT_FREE_MODE = "prev_close"  # prev_close(보수·채택) | after_close(당일 청산이 당일 시가 슬롯을 비움)
MIN_WIDTH = 20.0       # 박스 폭 하한 (%) — 2026-11-10 20~30% 채택
MAX_WIDTH = 30.0       # 박스 폭 상한 (%) — 기본 30%
ATR_LO = 0.0           # ATR 하한 (%) — 0=비활성
ATR_HI = 99.0          # ATR 상한 (%) — 99=비활성
SLIP = 0.0             # 슬리피지 (한쪽, 비율) — 진입 매수 +, 강제청산 매도 -
MIN_AMT = 0.0          # 진입 시 20일 평균 거래대금 하한 (통화) — 0=비활성
SLIP_MODE = "const"    # const | amt (거래대금 구간별 슬리피지)
SLIP_TIERS = [(1e8, 0.0003), (2e7, 0.0010), (5e6, 0.0025), (0.0, 0.0050)]
SELL_GAP = None        # 매도 간격 (비율) — None=매수 갭과 동일. 예: 0.0175
SIZE_MULT = [0.6, 0.6, 0.9, 1.2, 1.8]  # 차수별 매수 배수(1차부터, 10만 기준) — 후방배분 1:1:1.5:2:3
SIZE_SCORE = []        # 점수별 슬롯 배수 (0~7점 순). 예: 0.5,0.5,0.5,0.75,0.75,1,1.25,1.5
AMT_MIN = 1e9
START = 20140901       # 스캔 시작일 (fdr 과거 상한 2014-08)
BAND_DAYS = int(os.environ.get("BAND_DAYS", "0"))  # >0이면 일봉 N일 밴드로 실험
WEEKS = 6              # 주봉 밴드 길이(주) — 종가 6개 (2026-11-10 13주→6주)
BAND_SRC = "close"     # 밴드 재료: close(종가만) · mix(하단=저가, 상단=종가) · full(저가/고가)
COVER = 0.10           # 사다리 커버 → rungs = max(4, round(COVER/gap)) = 7차(갭 1.5%, 최심 -9.1%)
                       # 15차(-30%)는 박스 폭(22~30%) 밖: 5차 대비 순익 +6% · MDD +77% (2026-11-06 실측)
POOL = "w"             # 박스 풀: w=주봉박스만(현행) · wa=주봉+상승저점 · a=상승저점만
MAX_TRIPS = 5          # 당일 다중 왕복 상한 (sameday=multi — 같은 차수 재왕복 최대 횟수)
ENTRY_MODE = "close"   # 진입 시점: close(신호일 종가) · next_open(다음날 시가) · next_close(다음날 종가)
ENTRY_ZONE = False     # True면 진입 시점에도 매수존 조건 재확인
CANDLE_MODE = "bear"   # all | bear(음봉 신호만, 현행) | bear_tail(음봉+아래꼬리1%+)
WBREAK = "close"       # 주봉박스 소멸 판정: close | confirm2(2연속) | buffer1(1% 버퍼)
APOS = "far"           # A박스 위치 점수 방향: far(현행, 선에서 먼 쪽 가점) | near(가까운 쪽 가점)
ROTATE = "flat_first_excl"  # 무료 앱: 1차탈출 후 슬롯 종료·당일 재진입 금지 | keep=유료(슬롯 유지+재그리드)
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
        # 주봉 RSI14 (Wilder) — 박스 형성 이전의 과매도 여부 확인용
        _wd = np.diff(wclose, prepend=wclose[0])
        _up = pd.Series(np.where(_wd > 0, _wd, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        _dn = pd.Series(np.where(_wd < 0, -_wd, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        wrsi = (100 - 100 / (1 + _up / np.maximum(_dn, 1e-9))).to_numpy()
        wrsi_lo12_w = pd.Series(wrsi).rolling(12, min_periods=4).min().to_numpy()
        wend = np.array([int(d.strftime("%Y%m%d")) for d in wk.index],
                        dtype=np.int64)
        nw = len(wclose)
        wflagw = np.zeros(nw, dtype=bool)
        wlo_w = np.zeros(nw)
        whi_w = np.zeros(nw)
        wdriftw = np.zeros(nw)      # 13주 순드리프트
        wwidthw = np.zeros(nw)      # 밴드 폭 (최고/최저 - 1)
        wlow_a = (df["Low"].resample("W-FRI").min().reindex(wk.index)
                  .to_numpy(float) if BAND_SRC != "close" else wclose)
        whigh_a = (df["High"].resample("W-FRI").max().reindex(wk.index)
                   .to_numpy(float) if BAND_SRC == "full" else wclose)
        for k in range(WEEKS - 1, nw):
            seg = wclose[k - WEEKS + 1:k + 1]
            if BAND_SRC == "close":
                wlo_w[k], whi_w[k] = seg.min(), seg.max()
            else:
                wlo_w[k] = np.nanmin(wlow_a[k - WEEKS + 1:k + 1])
                whi_w[k] = np.nanmax(whigh_a[k - WEEKS + 1:k + 1])
            wdriftw[k] = seg[-1] / max(seg[0], 1e-9) - 1
            wwidthw[k] = seg.max() / max(seg.min(), 1e-9) - 1
            wflagw[k] = (wwidthw[k] <= MAX_WIDTH / 100 and abs(wdriftw[k]) <= 0.15)
        # 주봉 기간 — 밴드 조건 연속 유지 주 수
        wrun = np.zeros(nw, dtype=np.int64)
        run_w = 0
        for k in range(nw):
            run_w = run_w + 1 if wflagw[k] else 0
            wrun[k] = run_w
        kk = np.searchsorted(wend, dn) - 1
        okw = kk >= WEEKS - 1
        wdur = np.where(okw, wrun[np.clip(kk, 0, nw - 1)], 0).astype(np.int64)
        wflag = np.where(okw, wflagw[np.clip(kk, 0, nw - 1)], False)
        wlo = np.where(okw, wlo_w[np.clip(kk, 0, nw - 1)], 0.0)
        whi = np.where(okw, whi_w[np.clip(kk, 0, nw - 1)], 0.0)
        wdrift = np.where(okw, wdriftw[np.clip(kk, 0, nw - 1)], 0.0)
        wwidth = np.where(okw, wwidthw[np.clip(kk, 0, nw - 1)], 0.0)
        wpos = np.where(np.maximum(whi - wlo, 1e-9) > 0,
                        (c - wlo) / np.maximum(whi - wlo, 1e-9), 1.0)
        if BOX_SRC == "va":
            # ── 1년 거래대금 90% 혼잡 구간을 박스로 사용 (5일 간격 계산 후 채움)
            N_ = 250
            step = 5
            vlo_a = np.full(n, np.nan)
            vhi_a = np.full(n, np.nan)
            amt_ = c * v
            for e_ in range(N_ - 1, n, step):
                cs = c[e_ - N_ + 1:e_ + 1]
                ws = amt_[e_ - N_ + 1:e_ + 1]
                order = np.argsort(cs)
                csum = np.cumsum(ws[order])
                tot = csum[-1] if len(csum) else 0
                if tot <= 0:
                    continue
                lo_i = int(np.searchsorted(csum, tot * (1 - VA_PCT) / 2))
                hi_i = int(np.searchsorted(csum, tot * (1 + VA_PCT) / 2))
                vlo_a[e_] = cs[order][min(lo_i, N_ - 1)]
                vhi_a[e_] = cs[order][min(hi_i, N_ - 1)]
            # 앞으로 채우기
            last_lo = last_hi = np.nan
            for i_ in range(n):
                if not np.isnan(vlo_a[i_]):
                    last_lo, last_hi = vlo_a[i_], vhi_a[i_]
                else:
                    vlo_a[i_], vhi_a[i_] = last_lo, last_hi
            ok_v = ~np.isnan(vlo_a) & (vhi_a > vlo_a)
            wlo = np.where(ok_v, vlo_a, 0.0)
            whi = np.where(ok_v, vhi_a, 0.0)
            wwidth = np.where(ok_v, whi / np.maximum(wlo, 1e-9) - 1, 0.0)
            _ctr = (whi + wlo) / 2
            wdrift = np.where(ok_v, c / np.maximum(_ctr, 1e-9) - 1, 0.0)
            wflag = (ok_v & (wwidth >= MIN_WIDTH / 100) & (wwidth <= MAX_WIDTH / 100)
                     & (np.abs(wdrift) <= 0.25))
            wpos = np.where(np.maximum(whi - wlo, 1e-9) > 0,
                            (c - wlo) / np.maximum(whi - wlo, 1e-9), 1.0)
        if BAND_DAYS > 0:
            # 밴드를 주봉 13주 대신 일봉 N일 종가 밴드로 대체 (실험용)
            N = BAND_DAYS
            s_ = pd.Series(c)
            rlo = s_.rolling(N).min().values
            rhi = s_.rolling(N).max().values
            ok = np.arange(len(c)) >= N - 1
            wlo = np.where(ok, rlo, 0.0)
            whi = np.where(ok, rhi, 0.0)
            wwidth = np.where(ok, whi / np.maximum(wlo, 1e-9) - 1, 0.0)
            base = np.where(ok, s_.shift(N - 1).values, c)
            wdrift = np.where(ok, c / np.maximum(base, 1e-9) - 1, 0.0)
            wpos = np.where(np.maximum(whi - wlo, 1e-9) > 0,
                            (c - wlo) / np.maximum(whi - wlo, 1e-9), 1.0)
            okw = ok
            wflag = ok & (wwidth <= MAX_WIDTH / 100) & (np.abs(wdrift) <= 0.15)
            run_ = 0
            wdur = np.zeros(len(c), dtype=np.int64)
            for i_ in range(len(c)):
                run_ = run_ + 1 if wflag[i_] else 0
                wdur[i_] = run_
        okb = okw & (wwidth <= MAX_WIDTH / 100) & (np.abs(wdrift) <= 0.15)
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
        _amt_w = pd.Series(c * v).rolling(VP_WIN, min_periods=max(20, VP_WIN // 4)).sum().to_numpy()
        _vol_s = pd.Series(v).rolling(VP_WIN, min_periods=max(20, VP_WIN // 4)).sum().to_numpy()
        vwap250 = np.where(_vol_s > 0, _amt_w / np.maximum(_vol_s, 1e-9), np.nan)
        low52 = pd.Series(c).rolling(250, min_periods=60).min().to_numpy()
        # ── 눌림+박스 유지: 60일선 위 상승 추세 + 20일선 지지(저가 터치·종가 위) + 박스 유효
        ma20_ = pd.Series(c).rolling(20).mean().to_numpy()
        ma60_ = pd.Series(c).rolling(60).mean().to_numpy()
        _ma60_prev = np.concatenate([np.full(20, np.nan), ma60_[:-20]]) \
            if len(ma60_) > 20 else np.full(n, np.nan)
        pb_day = ((c > ma60_) & (ma60_ > _ma60_prev) & (ma20_ > 0)
                  & (l <= ma20_ * 1.01) & (c >= ma20_ * 0.99) & (c > o)
                  & wflag & (wwidth >= MIN_WIDTH / 100) & liq & (dn >= START))
        # ── 반전 돌파: 장기 하락 + 52주 바닥권 + 거래량 3배 + 20일 고점 돌파 + 3% 이상 상승
        _ma120 = pd.Series(c).rolling(120).mean().to_numpy()
        _ma120p = np.concatenate([np.full(20, np.nan), _ma120[:-20]])
        _v20 = pd.Series(v).rolling(20).mean().to_numpy()
        _low52 = pd.Series(c).rolling(250).min().to_numpy()
        _hi20 = pd.Series(h).shift(1).rolling(20).max().to_numpy()
        _ma20_ = pd.Series(c).rolling(20).mean().to_numpy()
        _ma20_prev = np.concatenate([[np.nan], _ma20_[:-1]])
        rv_day = ((c < _ma120) & (_ma120 < _ma120p)
                  & (c <= _low52 * 1.15)
                  & (v >= _v20 * 2.5)
                  & (c > _ma20_) & (_ma20_prev > 0)
                  & (np.concatenate([[False], c[:-1] <= _ma20_prev[1:]]))
                  & (c >= o * 1.03)
                  & liq & (dn >= START))
        wdays = np.where(wflag & (wpos <= 0.5) & liq & (dn >= START))[0]
        if POOL in ("wa", "w"):
            for j in wdays:
                kind_map.setdefault(int(j), "wbox")
        if PB_MODE:
            for j in np.where(pb_day)[0]:
                kind_map.setdefault(int(j), "wbox")
        if RV_MODE:
            for j in np.where(rv_day)[0]:
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
                        float(wdrift[j]), float(wwidth[j]),
                        float(wrsi[np.clip(kk[j], 0, len(wrsi) - 1)]))
            if knd == "abox":
                return (float(adist[j]) if not np.isnan(adist[j]) else None,
                        float(atr[j]) if not np.isnan(atr[j]) else None,
                        float(aslope[j]) if not np.isnan(aslope[j]) else None,
                        float(aheight[j]), None)
            return (None, None, None, None, None)
        sc = np.array([score_new(knd, *_comp(j, knd))
                       for j, knd in zip(ti, kinds_arr)], dtype=np.int64)
        if VP_SCORE and len(ti):
            _d = (wlo[ti] / np.maximum(vwap250[ti], 1e-9) - 1)
            _add = np.where(_d >= 0.15, 2, np.where(_d >= 0.05, 1,
                           np.where(_d >= -0.05, 0, -1)))
            sc = np.clip(sc + _add.astype(np.int64), 0, 7)
        U.append({"code": code, "name": name, "dn": dn, "o": o, "h": h, "l": l,
                  "c": c, "atr": atr, "amt20": amt20,
                  "bull": bull, "body": body, "lower_sh": lower_sh,
                  "wflag": wflag, "wvalid": wvalid, "wlo": wlo, "whi": whi, "wpos": wpos,
                  "wdur": wdur, "adur": adur, "wdrift": wdrift, "wwidth": wwidth,
                  "adist": adist, "aslope": aslope, "aheight": aheight,
                  "abox_ok": abox_ok, "abox_rise": abox_rise, "tlv": tlv,
                  "trig": ti, "kinds": kinds_arr, "sc": sc, "pb": pb_day, "ma20": ma20_, "rv": rv_day, "vwap": vwap250, "v": v, "low52": low52,
                "wrsi": wrsi[np.clip(kk, 0, nw - 1)],
                "wrsi_lo12": wrsi_lo12_w[np.clip(kk, 0, nw - 1)]})
    return U


SCHEME = "S4"   # 점수안: S4(수익 기준, 만점 7) · S2c(구) · S2b · S1 · S2 · S3


def score_new(kind, pos, atr, trend, height, rsi=None):
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
        if SCHEME == "S2c":       # ATR 3~7% +1 · 폭 2단(≥26% 2, ≥22% 1) → 만점 7
            a = 1 if (atr is not None and 0.03 <= atr < 0.07) else 0
            h = (2 if (height is not None and height >= 0.26)
                 else (1 if (height is not None and height >= 0.22) else 0))
            s = p + t + a + h
            if trend is not None and trend < -0.05:
                s -= 1
            return s
        if SCHEME == "S4":       # 수익 기준 (2026-11-13) — 위치 0~4 · ATR 0~2 · RSI +1 · 폭 +1 (만점 8)
            pp = (4 if (pos is not None and pos <= 0.24) else
                  3 if (pos is not None and pos <= 0.30) else
                  2 if (pos is not None and pos <= 0.35) else
                  1 if (pos is not None and pos <= 0.40) else 0)
            aa = 2 if (atr is not None and atr >= 0.07) else (
                 1 if (atr is not None and atr >= 0.055) else 0)
            rr = 1 if (rsi is not None and rsi >= 50) else 0
            hh = 1 if (height is not None and height >= 0.28) else 0
            return pp + aa + rr + hh
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
        sp = hst.get("slip", SLIP)
        if sp:
            px = px * (1 - sp)
        for k in range(len(hst["held"])):
            if hst["held"][k]:
                hst["net"] += ((px - hst["levels"][k]) / hst["levels"][k]
                               * UNIT * hst["size"][k])
                if k == 0 and hst["grid_n"] == 1:
                    hst["loss0"] = True
        done.append({"code": i, "symbol": hst.get("symbol"),
                     "name": hst.get("name"),
                     "entry_day": hst["entry_day"],
                     "exit_day": int(day), "score": hst["score"],
                     "escape0": hst["escape0"], "loss0": hst["loss0"],
                     "cycles": hst["cycles"], "net": hst["net"],
                     "days": hst["j"] - hst["j_entry"], "kind": why,
                     "ekind": hst.get("kind", "std"),
                     "dur": hst.get("dur", 0),
                     "cbull": hst.get("cbull"),
                     "cbody": hst.get("cbody"),
                     "clower": hst.get("clower"), "vdist": hst.get("vdist"),
                     "lo52d": hst.get("lo52d"), "wrsi": hst.get("wrsi"),
                     "wrsi_lo12": hst.get("wrsi_lo12")})
        if COMP_DUMP is not None:
            COMP_DUMP.append({
                "code": i, "symbol": hst.get("symbol"), "name": hst.get("name"),
                "exit_day": int(day),
                "entry_day": hst["entry_day"], "kind": hst.get("kind", "std"),
                "pos": hst.get("cpos"), "atr": hst.get("catr"),
                "trend": hst.get("ctrend"), "height": hst.get("cheight"),
                "score": hst["score"], "escape0": hst["escape0"],
                "loss0": hst["loss0"], "net": round(hst["net"]),
                "days": hst["j"] - hst["j_entry"], "cycles": hst["cycles"],
                "exit": why, "cbull": hst.get("cbull"),
                "cbody": hst.get("cbody"), "clower": hst.get("clower"), "vdist": hst.get("vdist"),
                     "lo52d": hst.get("lo52d"), "wrsi": hst.get("wrsi"),
                     "wrsi_lo12": hst.get("wrsi_lo12")})
        curve.append((int(day), hst["net"]))

    for D in all_days:
        if D < START:
            continue
        rotated_today = set()
        pending_exit = []
        free_prev = K_ - len(holdings)   # 당일 시작(전일 종가) 시점의 빈 슬롯
        for i in list(holdings):
            hst, s = holdings[i], U[i]
            dn = s["dn"]
            j = int(np.searchsorted(dn, D))
            if hst.get("exit_pending"):
                _px = float(s["o"][j]) if j < len(dn) else float(s["c"][hst["j"]])
                if _px > 0:
                    close_position(done, holdings, curve, i, _px, "하단근접(다음날시가)", D)
                    continue
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
            B_ = (hst.get("rv_line") or hst.get("pb_line")
                  or hst.get("ab_line")
                  or (wlo if knd == "wbox" else tlv))
            B_ = B_ if B_ > 0 else None
            if B_ is not None:
                lvl = B_ * (1 + EXIT_BUF / 100)
                if EXIT_MODE == "intraday":
                    o_j = float(s["o"][j])
                    fill = (o_j if (o_j > 0 and o_j < lvl)
                            else (lvl if lw < lvl else None))
                    if fill is not None:
                        close_position(done, holdings, curve, i, fill,
                                       "하단근접(장중)", D)
                        continue
                elif px < lvl:
                    if EXIT_AT == "next_open":
                        hst["exit_pending"] = True
                    else:
                        close_position(done, holdings, curve, i, px, "하단근접 청산", D)
                    continue
            # 1-b) 보유 기간 상한
            if TIME_STOP and (j - hst["j_entry"]) >= TIME_STOP:
                close_position(done, holdings, curve, i, px, "기간종료", D)
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
            stop_px = (B_ * (1 - HARD_STOP / 100)) if B_ is not None else None
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
                    if k < len(RUNG_FILL):
                        RUNG_FILL[k] += 1
            for k in range(len(held)):
                if not held[k]:
                    continue
                sg_ = SELL_GAP if SELL_GAP else g_i
                tgt = levels[k] * (1 + sg_)
                if float(s["h"][j]) < tgt:
                    continue
                if (k in bought) and sameday not in ("all", "multi"):
                    if sameday == "none":
                        continue
                    if sameday == "close" and px < tgt:
                        continue
                _sz = hst["size"][k]
                hst["net"] += (sg_ * UNIT - FEE * UNIT) * _sz
                held[k] = False
                hst["cycles"] += 1
                if k == 0 and hst["grid_n"] == 1:
                    hst["escape0"] = True
                # 같은 날 같은 차수 재왕복 — 당일 범위에서 2×갭 이동당 1회 추가
                # (저가 매수→고가 매도→재매수→재매도 경로 가정 · 분봉 다중 체결 근사)
                if sameday == "multi":
                    rng_d = float(s["h"][j]) - float(s["l"][j])
                    extra = int(rng_d // (2 * levels[k] * sg_))
                    extra = min(max(extra, 0), MAX_TRIPS)
                    if extra:
                        hst["net"] += extra * (sg_ * UNIT - FEE * UNIT) * _sz
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
                            _rm = (px / (_b * (1 + EXIT_BUF / 100)) - 1) * 100
                            nr = int(min(max(int(_rm // (g_i * 100)) + 1, 1), rungs))
                    hst["levels"] = [px * (1 - g_i) ** k for k in range(nr)]
                    hst["held"] = [False] * nr
                    hst["held"][0] = True
            held_cnt = sum(holdings[i]["held"])
            max_rungs = max(max_rungs, held_cnt)
        cap_daily.append(sum(
            sum(h["held"][k] * h["size"][k] for k in range(len(h["held"])))
            for h in holdings.values()) * UNIT)
        max_pos = max(max_pos, len(holdings))
        free = free_prev if SLOT_FREE_MODE == "prev_close" else K_ - len(holdings)
        if free > 0 and not (MKT_DN and int(D) in MKT_DN):
            for d, nsc, i, j, knd, _rp in ev_days.get(int(D), []):
                if free <= 0 or len(holdings) >= K_:
                    break
                if i in holdings:
                    continue
                if ROTATE in ("flat_first_excl", "flat_first_score") and i in rotated_today:
                    continue
                s = U[i]
                dn_s = s["dn"]
                pb_ok = PB_MODE and bool(s["pb"][j])
                rv_ok = RV_MODE and bool(s["rv"][j])
                if rv_ok:
                    pb_ok = False        # 반전 신호는 별도 처리
                if PB_SLOTS > 0 or RV_SLOTS > 0:
                    _npb = sum(1 for h in holdings.values() if h.get("pb"))
                    _nrv = sum(1 for h in holdings.values() if h.get("rv"))
                    _nbox = len(holdings) - _npb - _nrv
                    _cap = (RV_SLOTS if rv_ok else (PB_SLOTS if pb_ok else K_ - PB_SLOTS - RV_SLOTS))
                    _cur = _nrv if rv_ok else (_npb if pb_ok else _nbox)
                    if _cur >= _cap:
                        continue
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
                sl = SLIP
                if SLIP_MODE == "amt":
                    _a = float(s["amt20"][j])
                    sl = next((v for th, v in SLIP_TIERS if _a >= th), SLIP_TIERS[-1][1])
                if sl:
                    px_e = px_e * (1 + sl)
                if MIN_SCORE and -nsc < MIN_SCORE:
                    continue
                if MIN_AMT and float(s["amt20"][j]) < MIN_AMT:
                    continue
                if MIN_ROOM > 0 and knd == "wbox" and not (pb_ok or rv_ok):
                    _w0 = float(s["wlo"][je])
                    if _w0 > 0 and (px_e / (_w0 * (1 + EXIT_BUF / 100)) - 1) * 100 < MIN_ROOM:
                        continue
                if LO52_MIN > 0:
                    _l52 = float(s["low52"][je])
                    if not np.isnan(_l52) and _l52 > 0 and \
                            (px_e / _l52 - 1) * 100 < LO52_MIN:
                        continue
                if VP_MODE != "off" and knd == "wbox":
                    _vp = float(s["vwap"][j]) if not np.isnan(s["vwap"][j]) else 0.0
                    if _vp <= 0:
                        continue
                    _wlo_j = float(s["wlo"][je])
                    if not (_wlo_j > 0):
                        continue
                    _r = _wlo_j / _vp - 1
                    if VP_MODE == "below" and not (_r < 0):
                        continue
                    if VP_MODE == "above" and not (_r > 0):
                        continue
                    if VP_MODE == "near" and not (abs(_r) <= 0.03):
                        continue
                if MIN_WIDTH or ATR_LO or ATR_HI < 99:
                    _atr = float(s["atr"][j]) if not np.isnan(s["atr"][j]) else 0.0
                    if ATR_LO and _atr * 100 < ATR_LO:
                        continue
                    if ATR_HI < 99 and _atr * 100 > ATR_HI:
                        continue
                    if MIN_WIDTH and knd == "wbox" and float(s["wwidth"][je]) * 100 < MIN_WIDTH:
                        continue
                # 포지션별 갭 — ATR 모드: 신호일 ATR 기준(진입일 종가 미사용 = 룩어헤드 없음)
                if GAP_MODE == "atr":
                    a_sig = float(s["atr"][j])
                    gi = (min(max(ATR_K * a_sig, GAP_MIN), GAP_MAX)
                          if not np.isnan(a_sig) else gap)
                    n_i = NEAR_OF_GAP * gi * 100
                else:
                    gi, n_i = gap, near
                if CANDLE_MODE != "all" and not (pb_ok or rv_ok):   # 눌림/반전 신호는 자체 캔들 조건
                    if not (float(s["c"][j]) < float(s["o"][j])):
                        continue
                    if CANDLE_MODE == "bear_tail" \
                            and float(s["lower_sh"][j]) < 1.0:
                        continue
                if BOX_DUR and knd == "wbox":
                    if float(s["wdur"][je]) < BOX_DUR:
                        continue
                if VOL_MAX > 0:
                    _v20 = float(s["amt20"][j]) / max(float(s["c"][j]), 1e-9)
                    if _v20 > 0 and (float(s["v"][j]) if "v" in s else 0) / _v20 > VOL_MAX:
                        continue
                if POS_MAX < 50 and knd == "wbox" and not (pb_ok or rv_ok):
                    _w0p, _w1p = float(s["wlo"][je]), float(s["whi"][je])
                    if _w1p > _w0p > 0:
                        if (px_e - _w0p) / (_w1p - _w0p) * 100 > POS_MAX:
                            continue
                if ENTRY_ZONE and not (pb_ok or rv_ok):   # 눌림/반전 신호는 존 재확인 생략
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
                        _rm = (px_e / (_w0b * (1 + EXIT_BUF / 100)) - 1) * 100
                        ng = int(min(max(int(_rm // (gi * 100)) + 1, 1), rungs))
                _rv_line = (float(s["l"][j]) * 0.99 if rv_ok else None)
                _vp_boost = 1.0
                if knd == "wbox":
                    _vp_v = float(s["vwap"][j]) if not np.isnan(s["vwap"][j]) else 0.0
                    _wlo_e = float(s["wlo"][je])
                    if _vp_v > 0:
                        _d_e = _wlo_e / _vp_v - 1
                        if VP_TIER:
                            _vp_boost = (2.0 if _d_e >= 0.15 else
                                         1.5 if _d_e >= 0.05 else
                                         0.8 if _d_e < -0.05 else 1.0)
                        elif VP_BOOST != 1.0 and _wlo_e > _vp_v:
                            _vp_boost = VP_BOOST
                _pb_line = (float(s["ma20"][je])
                            if (pb_ok and PB_EXIT_LINE == "ma20"
                                and not np.isnan(s["ma20"][je])) else None)
                holdings[i] = {"symbol": s.get("code"), "name": s.get("name"),
                               "j": je, "j_entry": je,
                               "entry_day": int(dn_s[je]),
                               "entry": px_e, "score": -nsc, "pb": pb_ok, "rv": rv_ok,
                               "rv_line": _rv_line,
                               "ab_line": (float(s["tlv"][je])
                                           if (AB_PIN and knd == "abox"
                                               and not np.isnan(s["tlv"][je])) else None),
                               "pb_line": _pb_line,
                               "gap": gi, "near": n_i, "kind": knd,
                               "dur": (int(s["wdur"][je]) if knd == "wbox"
                                       else int(s["adur"][je]) if knd == "abox"
                                       else 0),
                               "cpos": (float(s["wpos"][je]) if knd == "wbox"
                                        else (float(s["adist"][je])
                                              if knd == "abox"
                                              and not np.isnan(s["adist"][je])
                                              else None)),
                               "wrsi": (float(s["wrsi"][je])
                                        if not np.isnan(s["wrsi"][je]) else None),
                               "wrsi_lo12": (float(s["wrsi_lo12"][je])
                                             if not np.isnan(s["wrsi_lo12"][je]) else None),
                               "lo52d": ((px_e / float(s["low52"][je]) - 1)
                                         if (not np.isnan(s["low52"][je])
                                             and s["low52"][je] > 0) else None),
                               "vdist": ((float(s["wlo"][je]) / float(s["vwap"][j]) - 1)
                                        if (not np.isnan(s["vwap"][j])
                                            and s["vwap"][j] > 0
                                            and float(s["wlo"][je]) > 0) else None),
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
                               "size": [x * (SIZE_SCORE[min(max(int(-nsc), 0), len(SIZE_SCORE) - 1)]
                                             if SIZE_SCORE else 1.0)
                                        * _vp_boost
                                        for x in (SIZE_MULT + [SIZE_MULT[-1]] * 8)[:ng]],
                               "slip": sl,
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
                                           "하단근접(당일)", int(dn_s[je]))
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
    bk = {"0~2": [], "3~4": [], "5~6": [], "7~8": []}
    for p in done:
        sc = p["score"]
        key = "7~8" if sc >= 7 else ("5~6" if sc >= 5 else ("3~4" if sc >= 3 else "0~2"))
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
        "rung_fill": RUNG_FILL[:8],
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
    global MIN_ROOM, EXIT_BUF, BAND_SRC, EXIT_AT, LO52_MIN, RANK
    global ADAPT_RUNGS
    global MIN_SCORE
    global K
    global MIN_WIDTH, MAX_WIDTH, ATR_LO, ATR_HI, SLIP, MIN_AMT, SELL_GAP, SIZE_MULT, PB_MODE, PB_EXIT_LINE, PB_SLOTS, RV_MODE, RV_SLOTS, AB_PIN, VP_MODE, VP_BOOST, VP_WIN, VP_SCORE, VP_TIER, BOX_SRC, VA_PCT, POS_MAX, BOX_DUR, TIME_STOP, VOL_MAX, HARD_STOP, SIZE_MULT_OVR
    ap = argparse.ArgumentParser(description="박스 Top-20 검증 시뮬레이션 (W/A 풀 · 하단근접 청산)")
    ap.add_argument("--sameday", choices=["close", "none", "all", "multi"],
                    default="all",
                    help="당일 체결 가정 (기본 all=저가체결·고가매도 · 분봉 실측 1.12배) · multi는 과대(1.88배)")
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
    ap.add_argument("--size-score", type=str, default="",
                    help="점수별 슬롯 배수 0~7점 (쉼표 구분). 비우면 균등")
    ap.add_argument("--band-days", type=int, default=BAND_DAYS)
    ap.add_argument("--weeks", type=int, default=WEEKS, help="주봉 밴드 길이(주), 기본 6")
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
    ap.add_argument("--rank", choices=["score", "pos"], default=RANK,
                    help="같은 날 후보 정렬: score=점수 높은 순(현행) · pos=밴드 하단 가까운 순")
    ap.add_argument("--lo52-min", type=float, default=LO52_MIN,
                    help="52주 저가 대비 최소 거리 %% — 이보다 가까우면 진입 제외 (기본 0)")
    ap.add_argument("--min-room", type=float, default=MIN_ROOM,
                    help="최소 진입 여지(%%) — 진입가→청산선 거리")
    ap.add_argument("--band-src", choices=["close", "mix", "full"], default=BAND_SRC,
                    help="밴드 재료: close=종가만(현행) · mix=하단 저가+상단 종가 · full=저가/고가")
    ap.add_argument("--exit-at", choices=["close", "next_open"], default=EXIT_AT,
                    help="청산 체결 시점: close=당일 종가 · next_open=다음날 시가")
    ap.add_argument("--exit-buf", type=float, default=EXIT_BUF,
                    help="청산선 위치(%%). +2=밴드하단 2%% 위(현행) · 0=하단 · -2=하단 2%% 아래")
    ap.add_argument("--adapt-rungs", action=argparse.BooleanOptionalAction,
                    default=ADAPT_RUNGS,
                    help="차수를 room에 맞춰 축소 (floor(room/gap)+1, 최대 5)")
    ap.add_argument("--min-score", type=int, default=MIN_SCORE,
                    help="최소 진입 점수 (0=비활성)")
    ap.add_argument("--slots", type=int, default=K,
                    help="동시 보유 슬롯 (기본 20 · 무료 매직스플릿 7)")
    ap.add_argument("--hard-stop", type=float, default=HARD_STOP,
                    help="경계 대비 하드스톱 %% (기본 15)")
    ap.add_argument("--box-dur", type=int, default=BOX_DUR,
                    help="박스 최소 유지 주 수")
    ap.add_argument("--time-stop", type=int, default=TIME_STOP,
                    help="보유 최대 거래일 (0=없음)")
    ap.add_argument("--vol-max", type=float, default=VOL_MAX,
                    help="신호일 거래량/20일평균 상한 (0=없음)")
    ap.add_argument("--pos-max", type=float, default=POS_MAX,
                    help="진입 시 밴드 위치 상한(%%)")
    ap.add_argument("--box-src", choices=["band", "va"], default=BOX_SRC,
                    help="박스 정의: band=주봉 6주(현행) · va=1년 거래대금 90%% 혼잡 구간")
    ap.add_argument("--va-pct", type=float, default=VA_PCT,
                    help="혼잡 구간 비율 (기본 0.90)")
    ap.add_argument("--vp-score", action="store_true",
                    help="매물대 거리로 점수 가감 (0.15↑ +2 · 0.05↑ +1 · -0.05↓ -1)")
    ap.add_argument("--vp-tier", action="store_true",
                    help="매물대 거리로 금액 차등 (0.15↑ ×2 · 0.05↑ ×1.5 · -0.05↓ ×0.8)")
    ap.add_argument("--vp-win", type=int, default=VP_WIN,
                    help="매물대 VWAP 기간(거래일), 기본 250")
    ap.add_argument("--vp-boost", type=float, default=VP_BOOST,
                    help="박스 하단이 매물대(250일 VWAP) 위일 때 슬롯 금액 배수")
    ap.add_argument("--vp", choices=["off", "below", "above", "near"], default=VP_MODE,
                    help="매물대 필터: below=박스하단이 250일 VWAP 아래 · above=위 · near=±3%%")
    ap.add_argument("--ab-pin", action="store_true",
                    help="A박스 추세선을 진입 시점 값으로 고정 (일별 재계산 대신)")
    ap.add_argument("--rv", action="store_true",
                    help="반전 돌파 진입 허용 (장기 하락 + 바닥 + 거래량 3배 + 20일 고점 돌파)")
    ap.add_argument("--rv-slots", type=int, default=RV_SLOTS,
                    help="반전 전용 슬롯 수 (0=구분 없음)")
    ap.add_argument("--pb-slots", type=int, default=PB_SLOTS,
                    help="눌림 전용 슬롯 수 (0=구분 없음)")
    ap.add_argument("--pb-exit", choices=["band", "ma20"], default=PB_EXIT_LINE,
                    help="눌림 포지션 청산선: band=밴드 하단 · ma20=진입 시점 20일선")
    ap.add_argument("--pb", action="store_true",
                    help="눌림+박스 유지 진입 허용 (20일선 지지·60일선 위)")
    ap.add_argument("--max-width", type=float, default=MAX_WIDTH,
                    help="박스 폭 상한(%%), 기본 30")
    ap.add_argument("--min-width", type=float, default=MIN_WIDTH,
                    help="박스 폭 하한(%%) — 예: 24")
    ap.add_argument("--atr-lo", type=float, default=ATR_LO, help="ATR 하한(%%)")
    ap.add_argument("--atr-hi", type=float, default=ATR_HI, help="ATR 상한(%%)")
    ap.add_argument("--slip", type=float, default=SLIP,
                    help="슬리피지(한쪽, %%) — 예: 0.3")
    ap.add_argument("--min-amt", type=float, default=MIN_AMT,
                    help="진입 시 20일 평균 거래대금 하한 (통화)")
    ap.add_argument("--sell-gap", type=float, default=0.0,
                    help="매도 간격(%%) — 0이면 매수 갭과 동일")
    ap.add_argument("--size-mult", type=str, default="",
                    help="차수별 매수 배수 (쉼표) — 예: 0.6,0.6,0.9,1.2,1.8")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--scheme", choices=["S1", "S2", "S2b", "S2c", "S3", "S4"],
                    default=SCHEME, help="점수안 (S4 기본=수익 기준 · 만점 7)")
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
                    default=ROTATE,
                    help="1차 탈출 후 슬롯 처리 (keep=재그리드 · flat_first=플랫 시 회전 · "
                         "flat_first_excl=회전 당일 동일종목 배제 · flat_first_score=점수 우위 시만)")
    ap.add_argument("--rotate-margin", type=float, default=2.0,
                    help="flat_first_score 점수 우위 기준 (기본 2점)")
    ap.add_argument("--wbreak", choices=["close", "confirm2", "buffer1"],
                    default="close",
                    help="주봉박스 소멸 판정 (close=일봉 즉시 · confirm2=2연속 · buffer1=1%% 버퍼)")
    ap.add_argument("--candle", choices=["all", "bear", "bear_tail"],
                    default="bear",
                    help="신호일 캔들 필터 (bear=음봉만 · bear_tail=음봉+아래꼬리1%%+)")
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
    globals()["BAND_DAYS"] = int(getattr(args, "band_days", 0) or 0)
    globals()["WEEKS"] = int(getattr(args, "weeks", 6) or 6)
    globals()["EXIT_BUF"] = float(getattr(args, "exit_buf", 2.0))
    globals()["EXIT_AT"] = getattr(args, "exit_at", "close")
    globals()["BAND_SRC"] = getattr(args, "band_src", "close")
    globals()["SIZE_SCORE"] = ([float(x) for x in args.size_score.split(",")]
                               if args.size_score.strip() else [])
    POOL = args.pool
    GAP_MODE = args.gap_mode
    ATR_K = args.atr_k
    NEAR_OF_GAP = args.near_of_gap
    GAP_MIN, GAP_MAX = args.gap_min, args.gap_max
    EXIT_MODE = args.exit_mode
    ENTRY_EVAL = args.entry_eval
    MIN_ROOM = args.min_room
    LO52_MIN = float(getattr(args, "lo52_min", 0.0))
    RANK = getattr(args, "rank", "score")
    ADAPT_RUNGS = args.adapt_rungs
    MIN_SCORE = args.min_score
    K = args.slots
    MIN_WIDTH = args.min_width
    MAX_WIDTH = args.max_width
    PB_MODE = bool(args.pb)
    PB_EXIT_LINE = args.pb_exit
    PB_SLOTS = max(0, int(args.pb_slots))
    RV_MODE = bool(args.rv)
    RV_SLOTS = max(0, int(args.rv_slots))
    AB_PIN = bool(args.ab_pin)
    VP_MODE = args.vp
    VP_BOOST = float(args.vp_boost)
    VP_WIN = max(20, int(args.vp_win))
    VP_SCORE = bool(args.vp_score)
    BOX_SRC = args.box_src
    POS_MAX = float(args.pos_max)
    BOX_DUR = max(0, int(args.box_dur))
    TIME_STOP = max(0, int(args.time_stop))
    VOL_MAX = float(args.vol_max)
    HARD_STOP = float(args.hard_stop)
    if args.size_mult.strip():
        SIZE_MULT_OVR = [float(x) for x in args.size_mult.split(",")]
        SIZE_MULT[:] = SIZE_MULT_OVR
    VA_PCT = float(args.va_pct)
    VP_TIER = bool(args.vp_tier)
    ATR_LO = args.atr_lo
    ATR_HI = args.atr_hi
    SLIP = args.slip / 100.0
    MIN_AMT = args.min_amt
    SELL_GAP = (args.sell_gap / 100.0) if args.sell_gap else None
    if args.size_mult:
        SIZE_MULT = [float(x) for x in args.size_mult.split(",")]
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
            _wp = float(s["wpos"][j]) if not np.isnan(s["wpos"][j]) else 1.0
            ev.append((int(s["dn"][j]), -int(sc), i, int(j), knd, _wp))
    if RANK == "pos":
        ev.sort(key=lambda e: (e[0], e[5], e[1], e[2]))
    else:
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
            subprocess.run(["git", "-C", str(REPO), "add", "data"], capture_output=True)
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
                   "weeks": WEEKS,
                   "min_width": MIN_WIDTH, "max_width": MAX_WIDTH,
                   "vp_boost": VP_BOOST, "vp_win": VP_WIN, "box_src": BOX_SRC,
                   "rotate": ROTATE,
                   "rotate_margin": ROT_MARGIN,
                   "pool": {"w": f"주봉박스({WEEKS}주 종가밴드)만",
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
            "min_room": MIN_ROOM, "scheme": SCHEME,
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
                                "candle_yearly", "rung_fill",
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
    subprocess.run(["git", "-C", str(REPO), "add", "data"], capture_output=True)
    subprocess.run(["git", "-C", str(REPO), "commit", "-m",
                    f"chore: 시뮬 갱신 (포지션 {s['positions']})"], capture_output=True)
    subprocess.run(["git", "-C", str(REPO), "push"], capture_output=True)
    print("[INFO] git push 완료")


if __name__ == "__main__":
    main()