"""
Retrieval evaluator for the local workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.config import get_settings
from app.workflow.engine import ResearchWorkflow


def load_eval_samples(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval dataset not found: {p}")

    samples: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = line.strip()
            if not row:
                continue
            try:
                obj = json.loads(row)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse failed at line={line_no}: {e}") from e

            for key in ("id", "query", "gold_chunk_ids"):
                if key not in obj:
                    raise ValueError(f"Missing field `{key}` at line={line_no}")
            if not isinstance(obj["gold_chunk_ids"], list):
                raise ValueError(f"`gold_chunk_ids` must be list at line={line_no}")

            samples.append(obj)

    if not samples:
        raise ValueError(f"Empty eval dataset: {p}")
    return samples


class EvalResult(BaseModel):
    hit_rate: float = 0.0
    recall_at_k: float = 0.0
    mrr_at_k: float = 0.0
    precision_at_k: float = 0.0
    sample_count: int = 0
    k: int = 0


class Evaluator:
    def __init__(self, workflow: ResearchWorkflow | None = None) -> None:
        self.workflow = workflow or ResearchWorkflow()

    @staticmethod
    def _metrics_for_one(
        retrieved_ids: list[str],
        gold_ids: list[str],
        k: int,
    ) -> tuple[float, float, float, float]:
        if k <= 0:
            raise ValueError("k must be > 0")

        top_ids = retrieved_ids[:k]
        gold_set = set(cid for cid in gold_ids if cid)
        if not gold_set:
            return 0.0, 0.0, 0.0, 0.0

        hit_ids = [cid for cid in top_ids if cid in gold_set]
        hit_rate = 1.0 if hit_ids else 0.0
        recall = len(set(hit_ids)) / len(gold_set)
        precision = len(hit_ids) / k

        mrr = 0.0
        for rank, cid in enumerate(top_ids, 1):
            if cid in gold_set:
                mrr = 1.0 / rank
                break
        return hit_rate, recall, mrr, precision

    def evaluate(self, path: str, k: int = 5) -> EvalResult:
        samples = load_eval_samples(path)
        total_hit = 0.0
        total_recall = 0.0
        total_mrr = 0.0
        total_precision = 0.0

        for idx, sample in enumerate(samples):
            query = str(sample["query"])
            gold_ids = list(sample["gold_chunk_ids"])
            sample_id = sample.get("id", f"row_{idx}")
            # Use isolated session to avoid history rewrite side effects in evaluation.
            eval_session_id = f"eval::{sample_id}"
            state = self.workflow.run(query, session_id=eval_session_id)
            retrieved_ids = [chunk.chunk_id for chunk in state.retrieved_chunks]

            hit, recall, mrr, precision = self._metrics_for_one(
                retrieved_ids=retrieved_ids,
                gold_ids=gold_ids,
                k=k,
            )
            total_hit += hit
            total_recall += recall
            total_mrr += mrr
            total_precision += precision

        n = len(samples)
        return EvalResult(
            hit_rate=round(total_hit / n, 4),
            recall_at_k=round(total_recall / n, 4),
            mrr_at_k=round(total_mrr / n, 4),
            precision_at_k=round(total_precision / n, 4),
            sample_count=n,
            k=k,
        )

    def evaluate_system(self, path: str | None = None, k: int | None = None) -> EvalResult:
        settings = get_settings()
        default_path = str(Path(settings.docs_dir).parent / "eval" / "sample_eval_questions.jsonl")
        eval_path = path or default_path
        eval_k = k if k is not None else 5
        return self.evaluate(path=eval_path, k=eval_k)


if __name__ == "__main__":
    evaluator = Evaluator()
    result = evaluator.evaluate_system()
    print(
        f"Hit Rate: {result.hit_rate:.2%}\n"
        f"Recall@{result.k}: {result.recall_at_k:.2%}\n"
        f"MRR@{result.k}: {result.mrr_at_k:.4f}\n"
        f"Precision@{result.k}: {result.precision_at_k:.2%}\n"
        f"Samples: {result.sample_count}"
    )
