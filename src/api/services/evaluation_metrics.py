"""evaluation_metrics.py: Formula-based live-run RAG metrics for Publishing Mode.

Notes:
- Live-run metrics below do NOT require labeled ground truth.
- Labeled retrieval metrics (Hit@1, Precision@k, Recall@k, MRR, NDCG)
  require relevant_chunk_ids and should be computed in offline eval scripts.
- Judge-based metrics (groundedness/faithfulness/etc.) are placeholders unless
  an explicit judge workflow is run.
"""

from __future__ import annotations

from typing import Literal


ModeType = Literal["ask_the_book", "positioning_brief", "marketing_copy"]


def _target_chunk_count(mode: ModeType) -> int:
    if mode == "ask_the_book":
        return 3
    if mode == "marketing_copy":
        return 4
    return 6  # positioning_brief


def _to_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _clip_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))


def _context_coverage_label(score: float) -> str:
    if score >= 0.80:
        return "strong"
    if score >= 0.55:
        return "moderate"
    if score >= 0.25:
        return "limited"
    return "weak"


def build_live_run_metrics(
    *,
    retrieved_chunks: list[object],
    mode: ModeType,
    output_type: str | None = None,
    missing_context_flags: list[str] | None = None,
    unsupported_claims_detected: bool | None = None,
    token_cost_metadata: dict[str, float | int | None] | None = None,
) -> dict[str, object]:
    """Compute formula-based live-run retrieval coverage + review signals."""
    scores: list[float] = []
    pages_or_sections: set[str] = set()

    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            score = _to_float(chunk.get("score"))
            page = chunk.get("page") or chunk.get("section") or chunk.get("chapter")
        else:
            score = _to_float(getattr(chunk, "score", None))
            page = getattr(chunk, "page", None) or getattr(chunk, "section", None) or getattr(chunk, "chapter", None)

        if score is not None:
            scores.append(score)
        if page is not None and str(page).strip():
            pages_or_sections.add(str(page).strip())

    retrieved_chunk_count = len(retrieved_chunks)
    avg_relevance_score = (sum(scores) / len(scores)) if scores else 0.0
    top_relevance_score = max(scores) if scores else 0.0

    target_count = _target_chunk_count(mode)
    chunk_count_score = min(retrieved_chunk_count / target_count, 1.0) if target_count > 0 else 0.0

    source_diversity_score: float | None
    if retrieved_chunk_count > 0 and pages_or_sections:
        source_diversity_score = len(pages_or_sections) / retrieved_chunk_count
    else:
        source_diversity_score = None

    normalized_avg_relevance_score = _clip_0_1(avg_relevance_score)

    if source_diversity_score is None:
        context_coverage_score = (
            0.55 * chunk_count_score
            + 0.45 * normalized_avg_relevance_score
        )
    else:
        context_coverage_score = (
            0.45 * chunk_count_score
            + 0.35 * normalized_avg_relevance_score
            + 0.20 * _clip_0_1(source_diversity_score)
        )
    context_coverage_score = _clip_0_1(context_coverage_score)
    context_coverage_label = _context_coverage_label(context_coverage_score)

    missing_context_present = bool(missing_context_flags)

    if unsupported_claims_detected is None:
        unsupported_claims_state = "unknown"
    else:
        unsupported_claims_state = "yes" if unsupported_claims_detected else "no"

    if unsupported_claims_detected is None:
        grounded_in_source = "unknown"
    elif retrieved_chunk_count <= 0:
        grounded_in_source = "no"
    elif unsupported_claims_detected is False:
        grounded_in_source = "yes"
    else:
        grounded_in_source = "no"

    below_target = retrieved_chunk_count < target_count
    human_review_recommended = (
        context_coverage_label in {"limited", "weak"}
        or below_target
        or bool(unsupported_claims_detected)
        or missing_context_present
        or mode in {"positioning_brief", "marketing_copy"}
    )

    workflow_trace = {
        "retrieved_chunk_count": retrieved_chunk_count,
        "selected_output": output_type or mode,
        "grounding_check": "retrieval_context_attached",
        "spoiler_setting": "not_set",
        "missing_context_flags_present": missing_context_present,
        "unsupported_claims_detected": unsupported_claims_state,
        "context_coverage_label": context_coverage_label,
    }

    quality_checks = {
        "grounded_in_source": grounded_in_source,
        "unsupported_claims_detected": unsupported_claims_state,
        "missing_context_present": missing_context_present,
        "human_review_recommended": human_review_recommended,
        "context_coverage_label": context_coverage_label,
        "context_coverage_score": context_coverage_score,
    }

    run_details_metrics: dict[str, object] = {
        "retrieved_chunk_count": retrieved_chunk_count,
        "avg_relevance_score": avg_relevance_score,
        "top_relevance_score": top_relevance_score,
        "chunk_count_score": chunk_count_score,
        "source_diversity_score": source_diversity_score,
        "normalized_avg_relevance_score": normalized_avg_relevance_score,
        "context_coverage_score": context_coverage_score,
        "context_coverage_label": context_coverage_label,
        # Advanced eval placeholders (require judge or labels)
        # Hit@1 / Precision@k / Recall@k / MRR / NDCG require relevant_chunk_ids.
        "hit_at_1": None,
        "precision_at_k": None,
        "recall_at_k": None,
        "mrr": None,
        "ndcg": None,
        # Judge-based placeholders (require LLM-as-judge or human annotation)
        "claim_support_coverage": None,
        "groundedness_score": None,
        "faithfulness_score": None,
        "unsupported_claim_count": None,
        "unsupported_claims": None,
        "answer_relevance": None,
        "answer_similarity": None,
    }

    if token_cost_metadata:
        run_details_metrics.update({
            "input_token_count": token_cost_metadata.get("input_token_count"),
            "output_token_count": token_cost_metadata.get("output_token_count"),
            "total_token_count": token_cost_metadata.get("total_token_count"),
            "estimated_cost_usd": token_cost_metadata.get("estimated_cost_usd"),
        })

    return {
        "workflow_trace": workflow_trace,
        "quality_checks": quality_checks,
        "run_details_metrics": run_details_metrics,
    }
