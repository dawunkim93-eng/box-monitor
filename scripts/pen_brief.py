"""연금 브리핑 — 시장 지표를 모아 이 에이전트(hermes -z)로 한국어 요약을 만들고
data/pen_brief.json 에 저장한다.

사용:
  pen_brief.py            # 수집 + LLM 요약
  pen_brief.py --no-llm   # 수집만 (프롬프트 확인용)
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import requests

OUT = "/home/lucky/box-monitor/data/pen_brief.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
KST = timezone(timedelta(hours=9))
SYMS = [
    ("^KS11", "코스피"), ("^KQ11", "코스닥"), ("^GSPC", "S&P500"),
    ("^IXIC", "나스닥"), ("KRW=X", "원달러"), ("^TNX", "미 10년물"),
    ("^VIX", "VIX"),
]


def series(sym, rng="6mo"):
    r = requests.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}",
                     params={"range": rng, "interval": "1d"}, headers=UA, timeout=20)
    q = r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    return [x for x in q if x is not None]


def pct(v, n):
    if len(v) <= n:
        return None
    return (v[-1] / v[-1 - n] - 1) * 100


def r2(x):
    return None if x is None else round(float(x), 2)


def collect():
    rows = []
    for sym, name in SYMS:
        try:
            v = series(sym)
            rows.append({
                "k": name, "v": round(v[-1], 2),
                "d1": r2(pct(v, 1)), "d5": r2(pct(v, 5)), "d20": r2(pct(v, 20)),
            })
        except Exception as e:
            rows.append({"k": name, "err": str(e)[:60]})
    return rows


def prompt_for(rows, asof):
    lines = []
    for r in rows:
        if "err" in r:
            continue
        d1 = f"{r['d1']:+.2f}%" if r.get("d1") is not None else "?"
        d5 = f"{r['d5']:+.2f}%" if r.get("d5") is not None else "?"
        d20 = f"{r['d20']:+.2f}%" if r.get("d20") is not None else "?"
        lines.append(f"- {r['k']}: {r['v']} (1일 {d1}, 5일 {d5}, 20일 {d20})")
    data = "\n".join(lines)
    return f"""너는 연금 계좌를 관리하는 사람에게 매일 아침 시장 상황을 3분 안에 읽히게 정리해 주는 조수다.
기준 시각: {asof} (한국시간)

시장 데이터:
{data}

아래 JSON 객체 하나만 출력해라. 설명, 인사, 코드블록 표시 금지.
{{"domestic": "국내 시장 2문장", "global": "해외 시장 2문장", "points": ["체크포인트 3개, 각 한 문장"], "caution": "오늘 특히 조심할 것 한 문장"}}

규칙:
- 한국어. 가운뎃점 기호(·) 대신 쉼표나 '와/과'를 쓴다.
- 과장하지 않는다. '폭등', '대박' 같은 표현 금지.
- 위 데이터에 없는 숫자를 만들지 않는다.
- 매수나 매도를 권하지 않는다. 상황과 확인할 것만 적는다.
- 각 항목은 짧게. 전문용어는 괄호로 풀어 쓴다."""


def ask_llm(prompt, timeout=240):
    p = subprocess.run(["hermes", "-z", prompt], capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "").strip()
    if p.returncode != 0:
        raise RuntimeError(f"hermes exit {p.returncode}: {(p.stderr or '')[:200]}")
    return out


def parse_json(txt):
    t = txt.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        t = t[i:j + 1]
    d = json.loads(t)
    for k in ("domestic", "global", "caution"):
        if not isinstance(d.get(k), str):
            raise ValueError(f"missing {k}")
    if not isinstance(d.get("points"), list):
        raise ValueError("missing points")
    d["points"] = [str(x) for x in d["points"]][:4]
    return d


def main():
    asof = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    rows = collect()
    pr = prompt_for(rows, asof)
    text = None
    err = None
    if "--no-llm" not in sys.argv:
        try:
            raw = ask_llm(pr)
            text = parse_json(raw)
        except Exception as e:
            err = str(e)[:300]
            text = {"domestic": "요약 생성 실패", "global": "", "points": [], "caution": ""}
    payload = {
        "asof": asof,
        "date": asof[:10],
        "market": rows,
        "text": text or {},
        "model": "hermes",
        "error": err,
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)
    print(asof, "브리핑 저장", "ok" if text and not err else f"err={err}")
    if text:
        print(text.get("domestic", ""))
        print(text.get("global", ""))


if __name__ == "__main__":
    sys.exit(main())
