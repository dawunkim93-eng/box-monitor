"""야후 5분봉 수집 (최근 55일) → data/min5/{code}.csv.gz

네이버 분봉은 9일치만 주므로, 진입 시각 검증용으로 5분봉 38거래일치를 모은다.
"""
import gzip
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/lucky/box-monitor")
OUT = ROOT / "data" / "min5"
UNIV = ROOT / "data" / "daily_universe.json"
MAP = ROOT / "data" / "market_map.json"
H = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
     "Accept": "application/json"}
DAYS = 55


def market_map():
    if MAP.exists():
        return json.loads(MAP.read_text(encoding="utf-8"))
    try:
        import FinanceDataReader as fdr
        out = {}
        for mk in ("KOSPI", "KOSDAQ"):
            df = fdr.StockListing(mk)
            for code in df["Code"].astype(str):
                out[code] = "KS" if mk == "KOSPI" else "KQ"
        MAP.write_text(json.dumps(out), encoding="utf-8")
        return out
    except Exception as e:
        print("[WARN] 시장 구분 실패", e)
        return {}


def fetch(sym, p1, p2):
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={p1}&period2={p2}&interval=5m")
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=25).read()
    d = json.loads(raw)
    r = (d.get("chart") or {}).get("result") or []
    if not r:
        return []
    r = r[0]
    ts = r.get("timestamp") or []
    q = r["indicators"]["quote"][0]
    cl = q.get("close") or []
    rows = []
    for t, c in zip(ts, cl):
        if c is None:
            continue
        k = time.strftime("%Y%m%d%H%M", time.localtime(t + 9 * 3600))
        rows.append((k, round(float(c), 2)))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    codes = [c for c in json.loads(UNIV.read_text(encoding="utf-8"))["codes"]
             if c[-1] not in "579K"]
    mm = market_map()
    p1 = int(time.time()) - DAYS * 86400
    p2 = int(time.time())
    ok = fail = skip = 0
    t0 = time.time()
    for n, code in enumerate(codes, 1):
        dest = OUT / f"{code}.csv.gz"
        if dest.exists() and dest.stat().st_size > 200:
            skip += 1
            continue
        suffix = mm.get(code)
        tries = [suffix] if suffix else ["KS", "KQ"]
        rows = []
        for sfx in tries:
            try:
                rows = fetch(f"{code}.{sfx}", p1, p2)
            except Exception:
                rows = []
            if rows:
                break
            time.sleep(0.15)
        if rows:
            with gzip.open(dest, "wt") as f:
                for k, c in rows:
                    f.write(f"{k},{c}\n")
            ok += 1
        else:
            fail += 1
        if n % 50 == 0:
            print(f"  ... {n}/{len(codes)} (성공 {ok} 실패 {fail} 스킵 {skip}, {time.time()-t0:.0f}s)",
                  flush=True)
        time.sleep(0.25)
    print(f"[INFO] 5분봉 수집 성공 {ok} · 실패 {fail} · 기존 {skip} → {OUT}")


if __name__ == "__main__":
    main()
