"""DART 재무제표 수집 — 분기별 매출액·영업이익 (2024~2026)

data/dart_fin.json : {종목코드: {"2024Q1": {"rev":..., "op":...}, ...}}
누적 기준(각 보고서의 당기 누적). 전년 동기와 비교해 개선 여부를 판단한다.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/lucky/box-monitor")
OUT = ROOT / "data" / "dart_fin.json"
CORP = ROOT / "data" / "dart_corp.json"
ENV = Path("/home/lucky/backtest/.env")
REPRT = {"11013": "Q1", "11012": "Q2", "11011": "Q4", "11014": "Q3"}
YEARS = [2024, 2025, 2026]


def key():
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("DART_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("DART_API_KEY", "")


def call(k, corp, year, rep, fs):
    url = ("https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
           f"?crtfc_key={k}&corp_code={corp}&bsns_year={year}"
           f"&reprt_code={rep}&fs_div={fs}")
    try:
        return json.loads(urllib.request.urlopen(url, timeout=25).read().decode("utf-8"))
    except Exception:
        return {"status": "err"}


def pick(d, want):
    for r in d.get("list", []):
        nm = (r.get("account_nm") or "").replace(" ", "")
        if nm in want:
            v = (r.get("thstrm_amount") or "").replace(",", "").strip()
            if v in ("", "-"):
                continue
            try:
                return int(v)
            except ValueError:
                continue
    return None


def main():
    k = key()
    cmap = json.loads(CORP.read_text(encoding="utf-8"))
    codes = json.loads((ROOT / "data" / "surge_codes.json").read_text(encoding="utf-8"))
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    t0 = time.time()
    n = 0
    for code in codes:
        if code in out and out[code]:
            continue
        info = cmap.get(code)
        if not info:
            out[code] = {}
            continue
        rec = {}
        for year in YEARS:
            for rep, tag in REPRT.items():
                d = call(k, info["corp_code"], year, rep, "CFS")
                if d.get("status") != "000":
                    d = call(k, info["corp_code"], year, rep, "OFS")
                if d.get("status") != "000":
                    continue
                rev = pick(d, {"매출액", "수익(매출액)", "매출", "영업수익"})
                op = pick(d, {"영업이익", "영업이익(손실)", "영업손실"})
                if rev is None and op is None:
                    continue
                rec[f"{year}{tag}"] = dict(rev=rev, op=op)
                time.sleep(0.04)
        out[code] = rec
        n += 1
        if n % 40 == 0:
            OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            print(f"  ... {n}/{len(codes)} ({time.time()-t0:.0f}s)", flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for v in out.values() if v)
    print(f"[INFO] 재무 수집 {len(out)}종목 (데이터 있음 {ok}) → {OUT.name}")


if __name__ == "__main__":
    main()
