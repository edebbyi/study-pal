"""evaluate_retrieval.py: Labeled retrieval evaluation scaffold for Publishing Mode.

This script computes retrieval metrics only when ground-truth relevant_chunk_ids
are available in the eval dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info < (3, 11):
    raise SystemExit(
        "StudyPal scripts require Python 3.11+. "
        "Run with your project env, for example: "
        "`myenv/bin/python scripts/evaluate_retrieval.py ...`"
    )

# Ensure repo root is importable when running as a standalone script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.document_workspace import (  # noqa: E402
    DocumentWorkspaceNotFoundError,
    load_workspace,
    normalize_user_id,
    retrieve_workspace_context,
    workspace_chunks,
    workspace_document_id,
    workspace_session_id,
    workspace_user_id,
)
from src.api.services.observability_service import (  # noqa: E402
    end_mlflow_run,
    log_mlflow_artifacts,
    log_mlflow_metrics,
    log_mlflow_params,
    start_mlflow_run,
) 
from src.core.config import settings  # noqa: E402
from src.data.retrieval import retrieve_chunks  # noqa: E402


@dataclass
class RetrievalEvalRow:
    doc_id: str
    question: str
    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str] | None = None


def _load_eval_rows(path: Path) -> list[RetrievalEvalRow]:
    rows: list[RetrievalEvalRow] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        doc_id = str(payload.get("doc_id") or "").strip()
        question = str(payload.get("question") or "").strip()
        relevant_chunk_ids = payload.get("relevant_chunk_ids")
        if not doc_id or not question or not isinstance(relevant_chunk_ids, list) or not relevant_chunk_ids:
            continue
        rows.append(
            RetrievalEvalRow(
                doc_id=doc_id,
                question=question,
                relevant_chunk_ids=[str(item) for item in relevant_chunk_ids if str(item).strip()],
                retrieved_chunk_ids=[
                    str(item) for item in payload.get("retrieved_chunk_ids", []) if str(item).strip()
                ]
                if isinstance(payload.get("retrieved_chunk_ids"), list)
                else None,
            )
        )
    return rows


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant)
    return hits / k


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant)
    return hits / len(relevant)


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def _rank_of_first_relevant(retrieved: list[str], relevant: set[str]) -> int | None:
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return index
    return None


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "hit_at_1": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "evaluated_rows": 0.0,
        }
    total = len(rows)
    return {
        "hit_at_1": sum(float(row.get("hit_at_1", 0.0) or 0.0) for row in rows) / total,
        "precision_at_k": sum(float(row.get("precision_at_k", 0.0) or 0.0) for row in rows) / total,
        "recall_at_k": sum(float(row.get("recall_at_k", 0.0) or 0.0) for row in rows) / total,
        "mrr": sum(float(row.get("mrr", 0.0) or 0.0) for row in rows) / total,
        "evaluated_rows": float(total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with labeled chunk ids.")
    parser.add_argument("--eval-file", default="evals/publishing_eval_sample.jsonl")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--k", type=int, default=max(settings.top_k, 5))
    parser.add_argument(
        "--compare-rerank",
        action="store_true",
        help=(
            "Run live retrieval twice per row (without and with rerank), "
            "then log rank-delta and metric deltas to MLflow."
        ),
    )
    parser.add_argument(
        "--candidate-pool-k",
        type=int,
        default=max(40, settings.rerank_candidates),
        help="Top-k candidate pool for --compare-rerank runs.",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        raise SystemExit(f"Eval file not found: {eval_path}")

    rows = _load_eval_rows(eval_path)
    if not rows:
        raise SystemExit("No valid labeled rows found. Ensure relevant_chunk_ids are provided.")

    normalized_user_id = normalize_user_id(args.user_id)
    k = max(1, int(args.k))

    run = start_mlflow_run(
        run_name="publishing.retrieval_eval",
        tags={"mode": "retrieval_eval", "k": str(k)},
    )
    log_mlflow_params(
        run,
        {
            "eval_file": str(eval_path),
            "k": k,
            "compare_rerank": bool(args.compare_rerank),
            "candidate_pool_k": int(max(1, args.candidate_pool_k)),
            "embedding_model": settings.embedding_model,
            "retrieval_algorithm": settings.retrieval_algorithm,
            "reranker_model": settings.rerank_model or "",
            "rerank_candidates": settings.rerank_candidates,
        },
    )

    per_row: list[dict[str, Any]] = []
    baseline_per_row: list[dict[str, Any]] = []
    reranked_per_row: list[dict[str, Any]] = []
    rank_delta_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []

    for row in rows:
        relevant_ids = set(row.relevant_chunk_ids)
        if args.compare_rerank:
            try:
                workspace = load_workspace(row.doc_id, normalized_user_id)
            except DocumentWorkspaceNotFoundError:
                skipped_rows.append(
                    {
                        "doc_id": row.doc_id,
                        "question": row.question,
                        "reason": "workspace_not_found",
                    }
                )
                continue
            resolved_user_id = workspace_user_id(workspace) or normalized_user_id
            raw_chunks = workspace_chunks(workspace)
            pre_retrieved = retrieve_chunks(
                question=row.question,
                chunks=raw_chunks,
                session_id=workspace_session_id(workspace),
                document_id=workspace_document_id(workspace),
                user_id=resolved_user_id,
                top_k=max(k, int(max(1, args.candidate_pool_k))),
                use_rerank=False,
            )
            post_retrieved = retrieve_chunks(
                question=row.question,
                chunks=raw_chunks,
                session_id=workspace_session_id(workspace),
                document_id=workspace_document_id(workspace),
                user_id=resolved_user_id,
                top_k=max(k, int(max(1, args.candidate_pool_k))),
                use_rerank=True,
            )
            pre_ids_full = [str(chunk.chunk_id) for chunk in pre_retrieved]
            post_ids_full = [str(chunk.chunk_id) for chunk in post_retrieved]
            pre_ids = pre_ids_full[:k]
            post_ids = post_ids_full[:k]

            pre_hit_at_1 = 1.0 if pre_ids and pre_ids[0] in relevant_ids else 0.0
            pre_precision_at_k = _precision_at_k(pre_ids, relevant_ids, k)
            pre_recall_at_k = _recall_at_k(pre_ids, relevant_ids, k)
            pre_mrr = _mrr(pre_ids, relevant_ids)

            post_hit_at_1 = 1.0 if post_ids and post_ids[0] in relevant_ids else 0.0
            post_precision_at_k = _precision_at_k(post_ids, relevant_ids, k)
            post_recall_at_k = _recall_at_k(post_ids, relevant_ids, k)
            post_mrr = _mrr(post_ids, relevant_ids)

            baseline_row = {
                "doc_id": row.doc_id,
                "question": row.question,
                "relevant_chunk_ids": row.relevant_chunk_ids,
                "retrieved_chunk_ids": pre_ids,
                "hit_at_1": pre_hit_at_1,
                "precision_at_k": pre_precision_at_k,
                "recall_at_k": pre_recall_at_k,
                "mrr": pre_mrr,
            }
            reranked_row = {
                "doc_id": row.doc_id,
                "question": row.question,
                "relevant_chunk_ids": row.relevant_chunk_ids,
                "retrieved_chunk_ids": post_ids,
                "hit_at_1": post_hit_at_1,
                "precision_at_k": post_precision_at_k,
                "recall_at_k": post_recall_at_k,
                "mrr": post_mrr,
            }
            baseline_per_row.append(baseline_row)
            reranked_per_row.append(reranked_row)

            pre_rank = _rank_of_first_relevant(pre_ids_full, relevant_ids)
            post_rank = _rank_of_first_relevant(post_ids_full, relevant_ids)
            rank_delta_rows.append(
                {
                    "doc_id": row.doc_id,
                    "question": row.question,
                    "relevant_chunk_ids": row.relevant_chunk_ids,
                    "pre_rerank_rank_of_first_relevant": pre_rank,
                    "post_rerank_rank_of_first_relevant": post_rank,
                    "rank_delta": (pre_rank - post_rank)
                    if pre_rank is not None and post_rank is not None
                    else None,
                    "top_candidates_pre": pre_ids_full[:20],
                    "top_candidates_post": post_ids_full[:20],
                }
            )

            # Main per-row output is reranked for top-level aggregate output.
            per_row.append(reranked_row)
        else:
            if row.retrieved_chunk_ids is not None:
                retrieved_ids = row.retrieved_chunk_ids[:k]
            else:
                try:
                    workspace = load_workspace(row.doc_id, normalized_user_id)
                except DocumentWorkspaceNotFoundError:
                    skipped_rows.append(
                        {
                            "doc_id": row.doc_id,
                            "question": row.question,
                            "reason": "workspace_not_found",
                        }
                    )
                    continue
                retrieved = retrieve_workspace_context(
                    workspace=workspace,
                    question=row.question,
                    user_id=normalized_user_id,
                    top_k=k,
                )
                retrieved_ids = [str(chunk.chunk_id) for chunk in retrieved]

            hit_at_1 = 1.0 if retrieved_ids and retrieved_ids[0] in relevant_ids else 0.0
            precision_at_k = _precision_at_k(retrieved_ids, relevant_ids, k)
            recall_at_k = _recall_at_k(retrieved_ids, relevant_ids, k)
            mrr = _mrr(retrieved_ids, relevant_ids)
            per_row.append(
                {
                    "doc_id": row.doc_id,
                    "question": row.question,
                    "relevant_chunk_ids": row.relevant_chunk_ids,
                    "retrieved_chunk_ids": retrieved_ids,
                    "hit_at_1": hit_at_1,
                    "precision_at_k": precision_at_k,
                    "recall_at_k": recall_at_k,
                    "mrr": mrr,
                }
            )

    aggregate = _aggregate_metrics(per_row)
    metric_payload: dict[str, float | int] = dict(aggregate)
    metric_payload["skipped_rows"] = len(skipped_rows)
    artifact_payload: dict[str, Any] = {
        "retrieval_eval_results": per_row,
        "retrieval_eval_aggregate": aggregate,
        "retrieval_eval_skipped_rows": skipped_rows,
    }

    if args.compare_rerank:
        baseline_aggregate = _aggregate_metrics(baseline_per_row)
        reranked_aggregate = _aggregate_metrics(reranked_per_row)
        metric_payload.update(
            {
                "baseline_hit_at_1": baseline_aggregate["hit_at_1"],
                "baseline_precision_at_k": baseline_aggregate["precision_at_k"],
                "baseline_recall_at_k": baseline_aggregate["recall_at_k"],
                "baseline_mrr": baseline_aggregate["mrr"],
                "reranked_hit_at_1": reranked_aggregate["hit_at_1"],
                "reranked_precision_at_k": reranked_aggregate["precision_at_k"],
                "reranked_recall_at_k": reranked_aggregate["recall_at_k"],
                "reranked_mrr": reranked_aggregate["mrr"],
                "delta_hit_at_1": reranked_aggregate["hit_at_1"] - baseline_aggregate["hit_at_1"],
                "delta_precision_at_k": reranked_aggregate["precision_at_k"] - baseline_aggregate["precision_at_k"],
                "delta_recall_at_k": reranked_aggregate["recall_at_k"] - baseline_aggregate["recall_at_k"],
                "delta_mrr": reranked_aggregate["mrr"] - baseline_aggregate["mrr"],
            }
        )
        artifact_payload.update(
            {
                "retrieval_eval_baseline_results": baseline_per_row,
                "retrieval_eval_baseline_aggregate": baseline_aggregate,
                "retrieval_eval_reranked_results": reranked_per_row,
                "retrieval_eval_reranked_aggregate": reranked_aggregate,
                "retrieval_eval_rerank_rank_deltas": rank_delta_rows,
            }
        )

    log_mlflow_metrics(run, metric_payload)
    log_mlflow_artifacts(run, artifact_payload)
    end_mlflow_run(run, status="FINISHED")

    print(json.dumps(metric_payload, indent=2))


if __name__ == "__main__":
    main()
