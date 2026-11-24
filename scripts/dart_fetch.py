"""DART 공시 데이터로 '재료의 성격' 구분 준비

1) corpCode.xml → 종목코드 ↔ DART 고유번호 매핑
2) 각 종목의 최근 2년 공시 목록 수집 → data/dart_disclosures.json
"""
import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path("/home/lucky/box-monitor")
OUT = ROOT / "data" / "dart_disclosures.json"
CORP = ROOT / "data" / "dart_corp.json"
ENV = Path("/home/lucky/backtest/.env")


def key():
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("DART_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("DART_API_KEY", "")


def corp_map(k):
    if CORP.exists():
        return json.loads(CORP.read_text(encoding="utf-8"))
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={k}"
    raw = urllib.request.urlopen(url, timeout=60).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    xml = zf.read(zf.namelist()[0]).decode("utf-8", "ignore")
    import re
    out = {}
    for m in re.finditer(r"<corp_code>(\d+)</corp_code>\s*<corp_name>([^<]*)</corp_name>"
                         r".*?<stock_code>(\d{6})</stock_code>", xml, re.S):
        out[m.group(3)] = dict(corp_code=m.group(1), name=m.group(2))
    CORP.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def list_disclosures(k, corp_code, bgn, end):
    rows = []
    page = 1
    while page <= 10:
        url = ("https://opendart.fss.or.kr/api/list.json"
               f"?crtfc_key={k}&corp_code={corp_code}"
               f"&bgn_de={bgn}&end_de={end}&page_no={page}&page_count=100")
        try:
            d = json.loads(urllib.request.urlopen(url, timeout=30).read().decode("utf-8"))
        except Exception:
            break
        if d.get("status") != "000":
            break
        rows += [dict(date=r["rcept_dt"], name=r["report_nm"]) for r in d.get("list", [])]
        if int(d.get("total_page", 1)) <= page:
            break
        page += 1
    return rows


def main():
    k = key()
    if not k:
        print("[ERR] DART_API_KEY 없음")
        return
    cmap = corp_map(k)
    codes = json.loads((ROOT / "data" / "surge_codes.json").read_text(encoding="utf-8"))
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    n = 0
    for code in codes:
        if code in out:
            continue
        info = cmap.get(code)
        if not info:
            out[code] = []
            continue
        rows = list_disclosures(k, info["corp_code"], "20240101", "20261130")
        out[code] = rows
        n += 1
        if n % 25 == 0:
            OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            print(f"  ... {n}종목")
        time.sleep(0.08)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    cnt = sum(len(v) for v in out.values())
    print(f"[INFO] 공시 수집 {len(out)}종목 · {cnt:,}건 → {OUT.name}")


if __name__ == "__main__":
    main()
