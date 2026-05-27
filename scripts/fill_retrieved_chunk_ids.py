"""fill_retrieved_chunk_ids.py: Batch-fill retrieved_chunk_ids for eval JSONL rows.

Example:
  python scripts/fill_retrieved_chunk_ids.py \
    --eval-file evals/publishing_eval_sample.jsonl \
    --doc-id c385eadd61 \
    --user-id debroah26@gmail.com \
    --k 10

By default, only rows with missing/empty retrieved_chunk_ids are updated.
Use --force to overwrite existing retrieved_chunk_ids.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from time import perf_counter
from pathlib import Path
from typing import Any

if sys.version_info < (3, 11):
    raise SystemExit(
        "StudyPal scripts require Python 3.11+. "
        "Run with your project env, for example: "
        "`myenv/bin/python scripts/fill_retrieved_chunk_ids.py ...`"
    )

# Ensure repo root is importable when running as a standalone script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.document_workspace import (  # noqa: E402
    DocumentRetrievalError,
    load_workspace,
    normalize_user_id,
    retrieve_workspace_context,
    workspace_chunks,
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"Invalid JSONL line {line_number}: expected object row.")
        rows.append(payload)
    return rows


def _should_update(row: dict[str, Any], *, force: bool) -> bool:
    if force:
        return True
    current = row.get("retrieved_chunk_ids")
    return not isinstance(current, list) or len(current) == 0


def _preview_question(text: str, limit: int = 110) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "what",
    "when",
    "where",
    "which",
    "who",
    "does",
    "have",
    "has",
    "was",
    "were",
    "are",
    "you",
    "your",
    "their",
    "them",
    "they",
    "will",
    "would",
    "about",
    "after",
    "before",
    "over",
    "under",
    "then",
    "than",
    "into",
    "onto",
    "across",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return {tok for tok in tokens if len(tok) >= 3 and tok not in _STOPWORDS}


def _overlap_score(question_terms: set[str], chunk_text: str) -> float:
    if not question_terms:
        return 0.0
    chunk_terms = _tokenize(chunk_text)
    if not chunk_terms:
        return 0.0
    overlap = len(question_terms & chunk_terms)
    if overlap == 0:
        return 0.0
    # Favors overlap while lightly normalizing by chunk vocabulary size.
    return overlap / max(8.0, float(len(chunk_terms)))


def _best_overlap_score(question: str, chunks: list[Any]) -> float:
    question_terms = _tokenize(question)
    best = 0.0
    for chunk in chunks:
        score = _overlap_score(question_terms, str(getattr(chunk, "text", "") or ""))
        if score > best:
            best = score
    return best


def _lexical_fallback_ids(*, workspace: dict[str, Any], question: str, k: int) -> list[str]:
    question_terms = _tokenize(question)
    ranked: list[tuple[float, str]] = []
    for chunk in workspace_chunks(workspace):
        chunk_text = str(getattr(chunk, "text", "") or "")
        score = _overlap_score(question_terms, chunk_text)
        ranked.append((score, str(chunk.chunk_id)))

    ranked.sort(key=lambda item: item[0], reverse=True)

    nonzero = [chunk_id for score, chunk_id in ranked if score > 0.0]
    if len(nonzero) >= k:
        return nonzero[:k]

    # If overlap is too sparse (OCR/noise), fall back to top-ranked regardless of zero score.
    all_ranked = [chunk_id for _score, chunk_id in ranked]
    return all_ranked[:k]


def _log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill retrieved_chunk_ids for eval JSONL rows.")
    parser.add_argument("--eval-file", default="evals/publishing_eval_sample.jsonl")
    parser.add_argument("--out-file", default=None, help="Optional output path; defaults to overwrite eval file.")
    parser.add_argument("--doc-id", default=None, help="Optional doc_id filter.")
    parser.add_argument("--user-id", default=None, help="Optional scoped user id for workspace loading.")
    parser.add_argument("--k", type=int, default=10, help="Top-k retrieved chunk ids to store.")
    parser.add_argument("--force", action="store_true", help="Overwrite retrieved_chunk_ids even if already set.")
    parser.add_argument(
        "--fallback-on-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When retrieval returns no grounded chunks, fill retrieved_chunk_ids with the first k "
            "workspace chunk ids and add a retrieval_warning (default: true)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-row runtime progress logs (default: true).",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        raise SystemExit(f"Eval file not found: {eval_path}")

    rows = _load_rows(eval_path)
    if not rows:
        raise SystemExit("No rows found in eval file.")

    normalized_user_id = normalize_user_id(args.user_id)
    k = max(1, int(args.k))
    workspace_cache: dict[str, dict[str, Any]] = {}
    workspace_error_cache: dict[str, str] = {}
    started_at = perf_counter()

    updated_count = 0
    skipped_count = 0
    error_count = 0

    _log(
        (
            f"[start] rows={len(rows)} file={eval_path} "
            f"doc_filter={args.doc_id or 'none'} top_k={k} force={args.force}"
        ),
        verbose=args.verbose,
    )

    for index, row in enumerate(rows, start=1):
        row_doc_id = str(row.get("doc_id") or "").strip()
        question = str(row.get("question") or "").strip()
        existing_ids = [
            str(item) for item in row.get("retrieved_chunk_ids", []) if str(item).strip()
        ] if isinstance(row.get("retrieved_chunk_ids"), list) else []
        _log(
            f"[row {index}/{len(rows)}] doc_id={row_doc_id or 'missing'} question={_preview_question(question)}",
            verbose=args.verbose,
        )
        if not row_doc_id or not question:
            _log("  -> skip: missing doc_id or question", verbose=args.verbose)
            skipped_count += 1
            continue
        if args.doc_id and row_doc_id != args.doc_id:
            _log(f"  -> skip: doc_id does not match filter {args.doc_id}", verbose=args.verbose)
            skipped_count += 1
            continue
        if not _should_update(row, force=args.force):
            _log("  -> skip: retrieved_chunk_ids already populated", verbose=args.verbose)
            skipped_count += 1
            continue

        workspace = workspace_cache.get(row_doc_id)
        if workspace is None:
            cached_workspace_error = workspace_error_cache.get(row_doc_id)
            if cached_workspace_error:
                row["retrieval_error"] = cached_workspace_error
                _log(f"  -> skip: cached workspace error ({cached_workspace_error})", verbose=args.verbose)
                error_count += 1
                continue
            try:
                _log("  -> loading workspace...", verbose=args.verbose)
                workspace = load_workspace(row_doc_id, normalized_user_id)
            except Exception as exc:
                detailed_error = f"workspace_load_failed: {type(exc).__name__}: {exc}"
                # Preserve pre-existing candidates when infra is unavailable.
                if existing_ids:
                    row["retrieval_warning"] = (
                        "workspace_unavailable_kept_existing_candidates: workspace lookup failed; "
                        "existing retrieved_chunk_ids were preserved."
                    )
                    _log(
                        "  -> warning: workspace unavailable; kept existing retrieved_chunk_ids",
                        verbose=args.verbose,
                    )
                    skipped_count += 1
                    workspace_error_cache[row_doc_id] = detailed_error
                    continue
                row["retrieval_error"] = detailed_error
                workspace_error_cache[row_doc_id] = detailed_error
                _log(f"  -> error: workspace load failed ({detailed_error})", verbose=args.verbose)
                error_count += 1
                continue
            workspace_cache[row_doc_id] = workspace
            _log("  -> workspace loaded", verbose=args.verbose)
        else:
            _log("  -> using cached workspace", verbose=args.verbose)

        try:
            _log("  -> retrieving chunks...", verbose=args.verbose)
            retrieved = retrieve_workspace_context(
                workspace=workspace,
                question=question,
                user_id=normalized_user_id,
                top_k=k,
            )
            retrieved_chunk_ids = [str(chunk.chunk_id) for chunk in retrieved]
            chosen_ids = retrieved_chunk_ids
            used_low_signal_fallback = False

            if args.fallback_on_empty and retrieved:
                # If semantic retrieval returns chunks with near-zero lexical overlap,
                # treat as low-signal and use lexical fallback candidates for labeling.
                best_overlap = _best_overlap_score(question, list(retrieved))
                if best_overlap <= 0.0:
                    chosen_ids = _lexical_fallback_ids(workspace=workspace, question=question, k=k)
                    used_low_signal_fallback = True

            row["retrieved_chunk_ids"] = chosen_ids
            # Keep file clean if row previously had an error.
            row.pop("retrieval_error", None)
            if used_low_signal_fallback:
                row["retrieval_warning"] = (
                    "retrieval_low_signal_fallback_used: semantic/local retrieval produced low-signal "
                    "candidates; ids were reseeded via lexical overlap for manual labeling."
                )
            else:
                row.pop("retrieval_warning", None)
            _log(
                (
                    f"  -> success: pulled {len(chosen_ids)} chunks | "
                    f"ids={chosen_ids}"
                ),
                verbose=args.verbose,
            )
            updated_count += 1
        except DocumentRetrievalError as exc:
            if args.fallback_on_empty:
                fallback_ids = _lexical_fallback_ids(workspace=workspace, question=question, k=k)
                row["retrieved_chunk_ids"] = fallback_ids
                row["retrieval_warning"] = (
                    "retrieval_empty_fallback_used: semantic/local retrieval returned no grounded chunks; "
                    "candidate ids were seeded via lexical overlap for manual labeling."
                )
                row["retrieval_error"] = f"retrieval_failed: {type(exc).__name__}: {exc}"
                _log(
                    (
                        "  -> warning: retrieval empty; seeded fallback ids="
                        f"{fallback_ids}"
                    ),
                    verbose=args.verbose,
                )
                updated_count += 1
                continue
            detailed_error = f"retrieval_failed: {type(exc).__name__}: {exc}"
            row["retrieval_error"] = detailed_error
            _log(f"  -> error: retrieval failed ({detailed_error})", verbose=args.verbose)
            error_count += 1
        except Exception as exc:
            detailed_error = f"retrieval_failed: {type(exc).__name__}: {exc}"
            row["retrieval_error"] = detailed_error
            _log(f"  -> error: retrieval failed ({detailed_error})", verbose=args.verbose)
            error_count += 1

    out_path = Path(args.out_file) if args.out_file else eval_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    duration_s = perf_counter() - started_at
    print(f"Rows updated: {updated_count}")
    print(f"Rows skipped: {skipped_count}")
    print(f"Rows errored: {error_count}")
    print(f"Runtime seconds: {duration_s:.2f}")
    print(f"Output written: {out_path}")


if __name__ == "__main__":
    main()
