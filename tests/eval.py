import csv
import json
from pathlib import Path
from typing import List

from config.logging_config import setup_logging
from config.settings import settings
from rag.retrieval import answer_question
from services.resources import get_llm, get_store


def load_eval_questions(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    setup_logging()
    store = get_store()
    llm = get_llm()
    eval_path = Path(__file__).parent / "sample_eval.csv"
    questions = load_eval_questions(eval_path)
    results = []
    for row in questions:
        q = row["question"]
        expected_source = row.get("expected_source", "")
        out = answer_question(q, store, llm)
        found = any(expected_source in cite.get("source", "") for cite in out["citations"])
        grounded = out["answer"].strip().lower() != "no information found."
        results.append({"question": q, "hit_expected": found, "grounded": grounded})
        print(f"Q: {q}")
        print(f"Answer: {out['answer']}")
        print(f"Hit expected: {found}\n")
    summary = {
        "total": len(results),
        "hit_rate": sum(1 for r in results if r["hit_expected"]) / max(len(results), 1),
        "grounded_rate": sum(1 for r in results if r["grounded"]) / max(len(results), 1),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
