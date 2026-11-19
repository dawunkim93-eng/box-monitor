"""저변동 모의매매 — 매일 후보 생성 + 결과 자동 채점

조건 (2년 검증 통과):
  A: 거래대금(20일) ≤ 37.5억 & 밴드위치(30일 종가밴드) ≥ 89.5%
  B: 거래대금(20일) ≤ 37.5억 & ATR ≤ 5.85%

거래 규칙:
  - 신호일 종가 = 지정가, 다음날 저가가 닿으면 체결
  - 3차 사다리: 지정가, -1.5%, -3.0% (6·6·9만원 = 슬롯 21만원)
  - 각 차수 +1.5% 터치 시 매도
  - 시간손절: 진입일 종가가 3차 가격 아래면 그날 종가 전량 청산
  - 최대 3거래일 보유 후 남은 물량 종가 청산

산출:
  data/paper_picks.json  — 누적 후보 (code, sig, limit, cond)
  data/daily_pick.json   — 오늘의 후보 + 성과 요약 + 최근 내역
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PIT = Path("/home/lucky/backtest/data/pit")
UNIV = ROOT / "data" / "daily_universe.json"
PICKS = ROOT / "data" / "paper_picks.json"
OUT = ROOT / "data" / "daily_pick.json"

GAP, FEE = 0.015, 0.00015
AMTS = [6, 6, 9]
SLOT = sum(AMTS) * 10_000
HOLD = 3
MIN_HIST = 60
MIN_AMT = 10.0      # 실전 최소 유동성(20일 평균 거래대금, 억원)
PREF_TAIL = "579K"  # 우선주(코드 끝자리) 제외
SURGE_MAX = 20.0    # 3일 수익률이 이보다 크면 제외 (급등 추격 방지)
COND = ROOT / "data" / "low_cond.json"


def cond_score(amt, pos, atr):
    """조건 점수 (유동성·밴드위치·ATR 칸 → 기대값 백분위 0~100)."""
    try:
        c = json.loads(COND.read_text(encoding="utf-8"))
    except Exception:
        return None
    bands = c["bands"]
    def idx(v, bs):
        for i, (lo, hi) in enumerate(bs):
            if lo <= v < hi:
                return i
        return None
    ai, pi, ti = idx(amt, bands["amt"]), idx(pos, bands["pos"]), idx(atr, bands["atr"])
    if None in (ai, pi, ti):
        return None
    alo, ahi = bands["amt"][ai]
    plo, phi = bands["pos"][pi]
    tlo, thi = bands["atr"][ti]
    key = (f"{alo:.0f}~{ahi:.0f}억",
           f"{plo:.0f}~{min(phi, 100):.0f}%",
           f"{tlo:.1f}~{min(thi, 20):.0f}%")
    for row in c["cells"]:
        if (row["amt"], row["pos"], row["atr"]) == key:
            return row["score"]
    return None


def naver_close(code, day=None):
    """네이버 일봉의 확정 종가 (day=YYYYMMDD 없으면 마지막 봉)."""
    import re
    import urllib.request
    try:
        end = pd.Timestamp.now()
        st = (end - pd.Timedelta(days=20)).strftime("%Y%m%d")
        url = ("https://api.finance.naver.com/siseJson.naver?"
               f"symbol={code}&requestType=1&startTime={st}&endTime={end:%Y%m%d}&timeframe=day")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", "ignore")
        rows = re.findall(r'\["(\d{8})",\s*[\d.]+,\s*[\d.]+,\s*[\d.]+,\s*([\d.]+)', raw)
        if not rows:
            return None
        if day is not None:
            for d, c in rows:
                if int(d) == int(day):
                    return float(c)
        return float(rows[-1][1])
    except Exception:
        return None


def naver_ohlc(code, day=None):
    """네이버 일봉에서 (시가, 고가, 저가, 종가) — day 없으면 마지막 봉."""
    import re
    import urllib.request
    try:
        end = pd.Timestamp.now()
        st = (end - pd.Timedelta(days=20)).strftime("%Y%m%d")
        url = ("https://api.finance.naver.com/siseJson.naver?"
               f"symbol={code}&requestType=1&startTime={st}&endTime={end:%Y%m%d}&timeframe=day")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", "ignore")
        rows = re.findall(r'\["(\d{8})",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', raw)
        if not rows:
            return None
        pick = rows[-1]
        if day is not None:
            for rr in rows:
                if int(rr[0]) == int(day):
                    pick = rr
                    break
        _, o, h, l, c = pick
        return float(o), float(h), float(l), float(c)
    except Exception:
        return None


def market_cooldown(sig_date, thr=-3.0):
    """신호일 KOSPI 등락 ≤ thr%이면 다음 거래일 신규 진입 중단. (True, pct) 반환."""
    import re
    import urllib.request
    if not sig_date:
        return False, None
    end = pd.Timestamp.now()
    st = (end - pd.Timedelta(days=30)).strftime("%Y%m%d")
    url = ("https://api.finance.naver.com/siseJson.naver?symbol=KOSPI"
           f"&requestType=1&startTime={st}&endTime={end:%Y%m%d}&timeframe=day")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", "ignore")
        rows = re.findall(r'\["(\d{8})",\s*[\d.]+,\s*[\d.]+,\s*[\d.]+,\s*([\d.]+)', raw)
        days = [int(d) for d, _ in rows]
        cl = [float(c) for _, c in rows]
        i = days.index(int(sig_date))
        if i == 0:
            return False, None
        pct = (cl[i] / cl[i - 1] - 1) * 100
        return pct <= thr, round(pct, 2)
    except Exception:
        return False, None


def load(code):
    try:
        df = pd.read_pickle(PIT / f"{code}.pkl")
    except Exception:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Volume"] > 0]
    return df if len(df) >= MIN_HIST else None


def features(df):
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c)
    tr = np.concatenate([[h[0] - l[0]],
                         np.maximum(h[1:] - l[1:],
                                    np.maximum(np.abs(h[1:] - c[:-1]),
                                               np.abs(l[1:] - c[:-1])))])
    atr = pd.Series(tr).ewm(alpha=1 / 14, min_periods=14).mean().values / np.maximum(c, 1e-9)
    cs = pd.Series(c)
    lo30 = cs.rolling(30).min().values
    hi30 = cs.rolling(30).max().values
    amt20 = pd.Series(c * v).rolling(20).mean().values / 1e8
    i = n - 1
    if lo30[i] <= 0 or hi30[i] <= lo30[i]:
        return None
    return dict(close=float(c[i]), open=float(df["Open"].values[i]),
                ret3=float((c[i] / c[i - 3] - 1) * 100) if i >= 3 else 0.0,
                prev_close=float(c[i - 1]) if i > 0 else float(c[i]),
                atr=float(atr[i] * 100), amt=float(amt20[i]),
                pos=float((c[i] - lo30[i]) / (hi30[i] - lo30[i]) * 100),
                lo30=float(lo30[i]), hi30=float(hi30[i]))


def evaluate(code, df, sig, limit):
    """신호일 종가 매수 기준 결과 계산."""
    days = [int(x.strftime("%Y%m%d")) for x in df.index]
    if sig not in days:
        return None
    i = days.index(sig)
    if i + 1 >= len(df):
        # 신호일 종가에 매수했고 아직 다음 거래일이 없음 → 보유중(평가 0)
        return dict(status="보유중", net=0.0, days=0, entry=sig,
                    exit=None, kind="보유중")
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    lv = [limit * (1 - GAP) ** k for k in range(3)]
    tg = [x * (1 + GAP) for x in lv]
    held = [True, False, False]    # 1차는 신호일 종가 매수
    net = 0.0
    exit_kind = None
    for dd in range(HOLD):
        j = i + 1 + dd
        if j >= len(df):
            break
        for k in range(1, 3):
            if not held[k] and float(l[j]) <= lv[k]:
                held[k] = True
        for k in range(3):
            if held[k] and float(h[j]) >= tg[k]:
                held[k] = False
                net += AMTS[k] * 10_000 * (GAP - FEE)
        if any(held) and float(c[j]) < lv[2]:      # 매일 종가 3차 이탈 → 그날 청산
            for k in range(3):
                if held[k]:
                    net += AMTS[k] * 10_000 * (float(c[j]) / lv[k] - 1 - FEE)
                    held[k] = False
            exit_kind = "손절선 청산"
            break
        if not any(held):
            exit_kind = exit_kind or "전량익절"
            break
    if any(held):
        full = i + HOLD                     # 기간청산 예정일
        if full <= len(df) - 1:             # 3거래일이 지났으면 종가 청산 확정
            for k in range(3):
                if held[k]:
                    net += AMTS[k] * 10_000 * (float(c[full]) / lv[k] - 1 - FEE)
            exit_kind = exit_kind or "만기 청산"
            exit_day = int(df.index[full].strftime("%Y%m%d"))
        else:                               # 아직 보유 중 — 평가손익만
            j = len(df) - 1
            for k in range(3):
                if held[k]:
                    net += AMTS[k] * 10_000 * (float(c[j]) / lv[k] - 1 - FEE)
            return dict(status="보유중", net=net / SLOT * 100, days=j - i,
                        entry=int(df.index[i + 1].strftime("%Y%m%d")),
                        exit=None, kind="보유중")
    else:
        exit_day = int(df.index[min(i + HOLD, len(df) - 1)].strftime("%Y%m%d"))
    return dict(status="청산완료", net=net / SLOT * 100, days=min(HOLD, len(df) - 1 - i),
                entry=int(df.index[i + 1].strftime("%Y%m%d")),
                exit=exit_day, kind=exit_kind)


def _market_close_min():
    """당일 장 마감 시각(분). data/market_hours.json 로 특수일 반영."""
    now = pd.Timestamp.now()
    key = now.strftime("%Y%m%d")
    try:
        cfg = json.loads((ROOT / "data" / "market_hours.json").read_text(encoding="utf-8"))
        v = cfg.get(key) or {}
        hh, mm = (v.get("close") or "15:30").split(":")
        return int(hh) * 60 + int(mm)
    except Exception:
        return 15 * 60 + 30


def _intraday_cut():
    """장 마감 +10분 전이면 오늘 봉은 미완성 — 신호일에서 제외."""
    now = pd.Timestamp.now()
    return now.hour * 60 + now.minute < _market_close_min() + 10, int(now.strftime("%Y%m%d"))


def fix_limits(picks, market_date):
    """모든 후보의 매수가를 신호일 확정 종가로 재확인."""
    fixed = 0
    for p in picks:
        d = load(p["code"])
        if d is None:
            continue
        days = [int(x.strftime("%Y%m%d")) for x in d.index]
        if p["sig"] not in days:
            continue
        real = float(d["Close"].values[days.index(p["sig"])])
        if p["sig"] == market_date:
            real = naver_close(p["code"], p["sig"]) or real
        if abs(real - float(p["limit"])) > 0.5:
            p["limit"] = round(real)
            fixed += 1
    return fixed


def main():
    codes = json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
    master = pd.read_pickle(PIT / "master.pkl")
    names = dict(zip(master["code"].astype(str), master["name"].astype(str)))

    # 시장 최신 '완료' 거래일 — 표본 종목의 최빈 마지막 날짜 (일부만 갱신된 경우 대비)
    from collections import Counter
    cut_intraday, today_int = _intraday_cut()
    _samp = []
    for _c in codes[:60]:
        d0 = load(_c)
        if d0 is None or not len(d0):
            continue
        if cut_intraday and int(d0.index[-1].strftime("%Y%m%d")) == today_int:
            d0 = d0.iloc[:-1]
        if len(d0):
            _samp.append(int(d0.index[-1].strftime("%Y%m%d")))
    market_date = Counter(_samp).most_common(1)[0][0] if _samp else None

    picks = json.loads(PICKS.read_text(encoding="utf-8")) if PICKS.exists() else []
    seen = {(p["code"], p["sig"]) for p in picks}
    today = None
    added = []
    scanned = []
    # 이미 보유 중(청산 전)인 종목은 신규 신호에서 제외
    open_codes = set()
    for _p in picks:
        _d = load(_p["code"])
        if _d is None:
            continue
        _r = evaluate(_p["code"], _d, _p["sig"], float(_p["limit"]))
        if _r and _r.get("status") == "보유중":
            open_codes.add(_p["code"])
    if open_codes:
        print(f"[INFO] 보유중 제외 {len(open_codes)}종목")

    cand = []
    surged = 0
    for code in codes:
        if code[-1] in PREF_TAIL:      # 우선주 제외
            continue
        _held = code in open_codes     # 보유중이면 표시만 하고 매수 안 함
        df = load(code)
        if df is None:
            continue
        if cut_intraday and int(df.index[-1].strftime("%Y%m%d")) == today_int:
            df = df.iloc[:-1]                  # 미완성 당일 봉 제외
        f = features(df)
        if not f:
            continue
        sig = int(df.index[-1].strftime("%Y%m%d"))
        scanned.append(sig)
        # PIT 종가가 확정 전 값일 수 있음 → 최신 거래일이면 네이버 확정가로 보정
        if sig == market_date:
            real = naver_close(code, sig)
            if real and abs(real - f["close"]) > 0.5:
                f["close"] = real
                lo, hi = min(f["lo30"], real), max(f["hi30"], real)
                f["pos"] = (real - lo) / max(hi - lo, 1e-9) * 100
        if f["ret3"] > SURGE_MAX:          # 급등 추격 제외
            surged += 1
            continue
        cond = []
        if f["amt"] >= MIN_AMT and f["amt"] <= 37.51 and f["pos"] >= 89.47:
            cond.append("A")
        if f["amt"] >= MIN_AMT and f["amt"] <= 37.51 and f["atr"] <= 5.85:
            cond.append("B")
        if "A" not in cond or (code, sig) in seen:   # 모의는 A(밴드 상단)만 추적
            continue
        cand.append(dict(code=code, name=names.get(code, code), sig=sig,
                         limit=round(f["close"]), cond="+".join(cond), held=_held,
                         amt=round(f["amt"], 1), pos=round(f["pos"], 1),
                         atr=round(f["atr"], 2),
                         open=round(f["open"]),
                         gap=round((f["open"] / max(f["prev_close"], 1e-9) - 1) * 100, 2),
                         score=cond_score(f["amt"], f["pos"], f["atr"])))

    today = Counter(scanned).most_common(1)[0][0] if scanned else market_date
    if surged:
        print(f"[INFO] 급등(3일 {SURGE_MAX:.0f}%↑) 제외 {surged}건")
    cool_on, cool_pct = market_cooldown(today)
    # 최신 거래일 신호만 등록 (거래정지·데이터 미갱신 종목의 옛 신호 배제)
    for rec in ([] if cool_on else cand):
        if rec["sig"] != today or (rec["code"], rec["sig"]) in seen:
            continue
        picks.append(rec)
        added.append(rec)
        seen.add((rec["code"], rec["sig"]))
    # 최근 30일 밖 신호는 정리
    cutoff = int((pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y%m%d"))
    picks = [p for p in picks if p["sig"] >= cutoff and p["code"][-1] not in PREF_TAIL]
    if cut_intraday:   # 장중 실행 시 당일(미완성) 후보 제거
        picks = [p for p in picks if p["sig"] != today_int]
    if cool_on:      # 쿨다운 날 신호는 원장에서 제외
        picks = [p for p in picks if p["sig"] != today]
    PICKS.write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")

    fixed = fix_limits(picks, market_date)
    if fixed:
        PICKS.write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[INFO] 지정가 보정 {fixed}건 (신호일 확정 종가)")

    rows = []
    for p in picks:
        df = load(p["code"])
        if df is None:
            continue
        r = evaluate(p["code"], df, p["sig"], float(p["limit"]))
        if r is None:
            r = dict(status="대기", net=0.0, days=0, entry=None, exit=None)
        rows.append({**p, **r})

    done = [r for r in rows if r["status"] == "청산완료"]
    fills = [r for r in rows if r["status"] not in ("미체결", "대기")]
    holding = [r for r in rows if r["status"] == "보유중"]
    tot = sum(r["net"] for r in done)
    summary = dict(
        picks=len(rows), entries=len(fills), filled=len(fills),
        done=len(done), holding=len(holding),
        fill_rate=round(len(fills) / max(len(rows), 1) * 100, 1),
        win=round(sum(1 for r in done if r["net"] > 0) / max(len(done), 1) * 100, 1),
        avg=round(tot / max(len(done), 1), 3), tot=round(tot, 1),
        period=dict(start=min((r["sig"] for r in fills), default=None),
                    end=max((r["sig"] for r in fills), default=None)))

    # 오늘 후보
    keys = ("code", "name", "limit", "cond", "amt", "pos", "atr", "score", "open", "gap")
    todays = [{k: r.get(k) for k in keys} for r in rows if r["sig"] == today]
    for t in todays:                      # 원장에 없는 값은 즉석 계산
        if t.get("score") is None:
            t["score"] = cond_score(t["amt"], t["pos"], t["atr"])
        if t.get("open") is None:
            _d = load(t["code"])
            _f = features(_d) if _d is not None else None
            if _f:
                t["open"] = round(_f["open"])
                t["gap"] = round((_f["open"] / max(_f["prev_close"], 1e-9) - 1) * 100, 2)
    if not todays and cool_on:     # 쿨다운 날은 참고용으로만 표시
        todays = [{k: c.get(k) for k in keys} for c in cand if c["sig"] == today]
    todays.sort(key=lambda t: ("A" not in t["cond"], -t["amt"]))
    for t in todays:
        t["rungs"] = [round(t["limit"] * (1 - GAP) ** k)
                      for k in range(3)]
        t["stop"] = round(t["limit"] * (1 - GAP) ** 2)

    # 청산 유형 분포 + 일별 누적 곡선
    kinds = {}
    for r in done:
        k = r.get("kind") or "기타"
        kinds[k] = kinds.get(k, 0) + 1
    byday = {}
    for r in done:   # 완료(실현) 기준 누적
        d0 = r.get("exit") or r["sig"]
        e = byday.setdefault(d0, dict(n=0, net=0.0))
        e["n"] += 1
        e["net"] += r["net"]
    curve = []
    cum = 0.0
    for d0 in sorted(byday):
        cum += byday[d0]["net"]
        curve.append(dict(day=d0, n=byday[d0]["n"],
                          net=round(byday[d0]["net"], 2), cum=round(cum, 2)))
    summary["kinds"] = kinds
    summary["days"] = len(curve)
    summary["best"] = round(max([r["net"] for r in done], default=0), 2)
    summary["worst"] = round(min([r["net"] for r in done], default=0), 2)

    # 최근 3거래일 (신호가 없던 날도 빈 칸으로 표시)
    _mk0 = pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"]).set_index("Date")
    _cal = [int(d.strftime("%Y%m%d")) for d in _mk0.index
            if int(d.strftime("%Y%m%d")) <= (today or 0)]
    _base_sigs = list(_cal[-3:][::-1]) if _cal else []
    # 아직 보유중인 포지션이 있으면 그 신호일도 포함 (오래돼도 표시)
    _open_sigs = sorted({r["sig"] for r in rows if r.get("status") == "보유중"},
                        reverse=True)
    _sigs = sorted(set(_base_sigs) | set(_open_sigs), reverse=True)[:6]
    # 쿨다운 차단일 (사유 표기용)
    _mk = pd.read_csv(ROOT / "data" / "ks11.csv", parse_dates=["Date"]).set_index("Date")
    _md = [int(d.strftime("%Y%m%d")) for d in _mk.index]
    _mr = _mk["Close"].pct_change() * 100
    _blocked = {_md[i] for i in range(1, len(_md)) if _mr.iloc[i] <= -3.0}
    days_out = []
    for _sg in _sigs:
        items = []
        _extra = []
        if _sg == today:
            _extra = [dict(code=c["code"], name=c["name"], sig=c["sig"], limit=c["limit"],
                           cond=c.get("cond"), amt=c.get("amt"), pos=c.get("pos"),
                           atr=c.get("atr"), score=c.get("score"), held=True)
                      for c in cand if c["sig"] == _sg and c.get("held")]
        _seen_codes = {x["code"] for x in picks if x["sig"] == _sg}
        _extra = [c for c in _extra if c["code"] not in _seen_codes]
        for p in [x for x in picks if x["sig"] == _sg] + _extra:
            r = next((x for x in rows if x["code"] == p["code"] and x["sig"] == _sg), None)
            d0 = load(p["code"])
            op = gp = None
            if d0 is not None:
                ds = [int(x.strftime("%Y%m%d")) for x in d0.index]
                if _sg in ds:
                    i0 = ds.index(_sg)
                    if i0 + 1 < len(d0):
                        op = float(d0["Open"].values[i0 + 1])
                        gp = round((op / float(p["limit"]) - 1) * 100, 2)
            _st = round(float(p["limit"]) * 0.970225)
            _cur = _hold = _under = None
            _dstat = _dkinds = None
            _today_real = int(pd.Timestamp.now().strftime("%Y%m%d"))
            _entry = None
            if d0 is not None:
                ds2 = [int(x.strftime("%Y%m%d")) for x in d0.index]
                if _sg in ds2:
                    _i2 = ds2.index(_sg)
                    if _i2 + 1 < len(d0):
                        _entry = ds2[_i2 + 1]
                        _hold = (len(d0) - (_i2 + 1)) + 1
                    elif _today_real > _sg:
                        _entry = _today_real      # PIT에 오늘 봉이 아직 없음
                        _hold = 1
            # 진입일이 지났으면(오늘 포함) 현재가로 손절선 이탈 판정
            if _entry is not None and _entry <= _today_real:
                _cur = naver_close(p["code"], _today_real) or (
                    float(d0["Close"].values[-1]) if d0 is not None and len(d0) else None)
                _under = bool(_cur and _cur < _st)
                # 오늘 봉이 PIT에 없으면 장중 고저로 청산 여부까지 확인
                if _entry == _today_real:
                    _bar = naver_ohlc(p["code"], _today_real)
                    if _bar:
                        _o, _h, _l, _c = _bar
                        _lim = float(p["limit"])
                        _lv = [_lim * 0.985 ** k for k in range(3)]
                        _held = {}
                        _filled = _sold = 0
                        if _l <= _lim:
                            _held[0] = min(_o, _lim)
                            _filled = 1
                            for _k in (1, 2):
                                if _l <= _lv[_k]:
                                    _held[_k] = _lv[_k]
                                    _filled += 1
                            # 갭상승 출발(시가>지정가)이면 고가가 시가일 수 있어
                            # 종가가 목표를 넘긴 경우만 당일 매도 인정
                            for _k in list(_held):
                                _tgt = _held[_k] * 1.015
                                _ok = (_c >= _tgt) if _o > _lim else (_h >= _tgt)
                                if _ok:
                                    del _held[_k]
                                    _sold += 1
                        _cur = _c
                        _under = bool(_held and _c < _lv[2])
                        # 오늘 상태 판정 (PIT 지연 보완)
                        if not _filled:
                            _dstat = "미체결"
                        elif not _held:
                            _dstat = "청산완료"
                        else:
                            _dstat = "보유중"
                        _dkinds = ("전량익절" if (not _held and _sold == _filled)
                                   else ("손절선 청산" if _under else None))
            items.append(dict(code=p["code"], name=p["name"], limit=p["limit"],
                              stop=_st, cur=(round(_cur) if _cur else None),
                              under=_under, hold=_hold,
                              score=p.get("score"), cond=p.get("cond"),
                              amt=p.get("amt"), pos=p.get("pos"), atr=p.get("atr"),
                              open=(round(op) if op else None), gap=gp,
                              held=bool(p.get("held")),
                              status=("보유중" if p.get("held") else
                                      ("매수 예정" if _sg == today and not _dstat else
                                       (_dstat or (r or {}).get("status", "대기")))),
                              kind=(_dkinds or (r or {}).get("kind")),
                              net=round((r or {}).get("net", 0), 2)))
        items.sort(key=lambda x: (-(x["score"] or 0)))
        _why = ("급락 쿨다운" if _sg in _blocked else
                ("조건 미달" if not items else ""))
        days_out.append(dict(sig=_sg, n=len(items), items=items, why=_why))

    # 오늘 청산된 종목 (표 음영용)
    _td = int(pd.Timestamp.now().strftime("%Y%m%d"))
    _ct = {}
    for r in done:
        if r.get("exit") != _td:
            continue
        e = _ct.setdefault(r["code"], dict(code=r["code"], name=r["name"], net=0.0,
                                           kinds=[], exit=r.get("exit")))
        e["net"] = round(e["net"] + r["net"], 2)
        if r.get("kind"):
            e["kinds"].append(r["kind"])
    closed_today = list(_ct.values())

    out = dict(updated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
               closed_today=closed_today,
               days=days_out,
               cooldown=dict(on=bool(cool_on), pct=cool_pct, thr=-3.0),
               curve=curve[-40:],
               sig_date=today, summary=summary,
               today=todays,
               recent=sorted(rows, key=lambda r: -r["sig"])[:12],
               rule=dict(gap=GAP * 100, amts=AMTS, slot=SLOT, hold=HOLD,
                         cond_a="거래대금 ≤ 37.5억 & 밴드위치 ≥ 89.5%",
                         cond_b="거래대금 ≤ 37.5억 & ATR ≤ 5.85%",
                         min_amt=MIN_AMT,
                         entry="전일 종가 지정가", stop="종가가 3차 가격(-3%) 아래면 그날 청산 (1~3일차 매일 적용)"))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if cool_on:
        print(f"[INFO] ⛔ 급락 쿨다운 — 신호일 KOSPI {cool_pct:+.2f}% · 신규 진입 중단")
    print(f"[INFO] 신호일 {today} · 오늘 후보 {len(todays)}건 · 신규 {len(added)}건 · "
          f"누적 {len(rows)}건(체결 {len(fills)}) · 완료 {len(done)}건 "
          f"승률 {summary['win']}% 평균 {summary['avg']:+.3f}% 합계 {summary['tot']:+.1f}%")
    for t in todays:
        print(f"   {t['name']:<12} {t['cond']} 지정가 {t['limit']:>8,.0f} "
              f"차수 {t['rungs']} 손절선 {t['stop']:>8,.0f} · 거래대금 {t['amt']}억")


if __name__ == "__main__":
    main()
