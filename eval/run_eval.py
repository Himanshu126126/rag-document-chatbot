"""Small evaluation: retrieval hit rate and answer keyword match.

Usage (from project root):
    python -m eval.run_eval --pdf data/sample.pdf --cases eval/eval_set.json
"""
import argparse
import json
from pathlib import Path

from src.chain import ask
from src.ingest import build_index


def run(pdf: str, cases: str) -> None:
    if not Path(pdf).exists():
        raise SystemExit(f"PDF not found: {pdf}. Put your test PDF in data/.")
    store = build_index([Path(pdf)])
    tests = json.loads(Path(cases).read_text(encoding="utf-8"))

    hits = correct = 0
    for t in tests:
        answer, docs = ask(store, t["question"])
        context = " ".join(d.page_content.lower() for d in docs)
        keywords = [k.lower() for k in t["keywords"]]
        hit = all(k in context for k in keywords)
        match = all(k in answer.lower() for k in keywords)
        hits += hit
        correct += match
        print(f"[{'OK' if match else 'MISS'}] retrieval={hit} | {t['question']}")

    n = len(tests)
    print(f"\nRetrieval hit rate  : {hits}/{n} = {hits / n:.0%}")
    print(f"Answer keyword match: {correct}/{n} = {correct / n:.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data/sample.pdf")
    parser.add_argument("--cases", default="eval/eval_set.json")
    args = parser.parse_args()
    run(args.pdf, args.cases)
