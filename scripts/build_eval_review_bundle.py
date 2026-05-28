"""build_eval_review_bundle.py: Build a human-review set for retrieval labeling.

This script reads eval JSONL rows and materializes question-level candidate chunks
with full text into:
1) a Markdown review document, and
2) a JSON artifact for programmatic use.

Example:
  myenv/bin/python scripts/build_eval_review_bundle.py \
    --eval-file evals/publishing_eval_sample.jsonl \
    --doc-id c385eadd61 \
    --out-md evals/review_bundle_c385eadd61.md \
    --out-json evals/review_bundle_c385eadd61.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

if sys.version_info < (3, 11):
    raise SystemExit(
        "StudyPal scripts require Python 3.11+. "
        "Run with your project env, for example: "
        "`myenv/bin/python scripts/build_eval_review_bundle.py ...`"
    )

# Ensure repo root is importable when running as a standalone script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.document_workspace import load_workspace, normalize_user_id, workspace_chunks  # noqa: E402


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


def _rank_chunk_ids_by_lexical_overlap(
    *,
    chunks: list[dict[str, Any]],
    query_text: str,
    limit: int,
) -> list[str]:
    query_terms = _tokenize(query_text)
    if not query_terms:
        return [str(chunk.get("chunk_id") or "") for chunk in chunks][:limit]

    # Query-term IDF across chunks, so rare query terms (e.g., "cyclone", "toto")
    # matter more than common ones (e.g., "dorothy").
    n_chunks = max(1, len(chunks))
    df: dict[str, int] = {term: 0 for term in query_terms}
    chunk_term_cache: dict[str, set[str]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        terms = _tokenize(str(chunk.get("text") or ""))
        chunk_term_cache[chunk_id] = terms
        for term in query_terms:
            if term in terms:
                df[term] += 1
    idf: dict[str, float] = {
        term: (math.log((n_chunks + 1.0) / (df_count + 1.0)) + 1.0)
        for term, df_count in df.items()
    }

    ranked: list[tuple[float, str]] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        chunk_terms = chunk_term_cache.get(chunk_id, set())
        overlap_terms = query_terms & chunk_terms
        chunk_text_lower = str(chunk.get("text") or "").lower()
        substring_terms = {
            term
            for term in query_terms
            if len(term) >= 5 and term in chunk_text_lower
        }
        matched_terms = overlap_terms | substring_terms
        if not matched_terms:
            score = 0.0
        else:
            weighted_overlap = sum(idf.get(term, 1.0) for term in matched_terms)
            # Light length normalization to avoid overweighting tiny noisy chunks.
            score = weighted_overlap / max(4.0, math.sqrt(float(len(chunk_terms) or 1)))
            # One-token overlap is usually noisy for OCR-heavy corpora.
            if len(matched_terms) < 2:
                score *= 0.15
        ranked.append((score, chunk_id))
    ranked.sort(key=lambda item: item[0], reverse=True)

    nonzero = [chunk_id for score, chunk_id in ranked if score > 0.0]
    if len(nonzero) >= limit:
        return nonzero[:limit]

    return [chunk_id for _score, chunk_id in ranked][:limit]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"Invalid JSONL line {line_number}: expected object row.")
        rows.append(row)
    return rows


def _load_chunks_from_catalog(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Catalog file must be a JSON object.")
    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list):
        raise SystemExit("Catalog file is missing a 'chunks' list.")
    parsed: list[dict[str, Any]] = []
    for item in raw_chunks:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        parsed.append(
            {
                "chunk_id": chunk_id,
                "page": item.get("page"),
                "chapter": item.get("chapter"),
                "topic": item.get("topic"),
                "citation": item.get("citation"),
                "text": str(item.get("text") or ""),
            }
        )
    return parsed, str(payload.get("document_title") or "").strip() or None


def _catalog_payload_from_workspace(*, doc_id: str, workspace: dict[str, Any], chunks: list[Any]) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "document_title": str(workspace.get("document_title") or "").strip() or None,
        "filename": str(workspace.get("filename") or "").strip() or None,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": str(chunk.chunk_id),
                "page": chunk.page,
                "chapter": chunk.chapter,
                "topic": chunk.topic,
                "citation": chunk.citation,
                "text": chunk.text,
            }
            for chunk in chunks
        ],
    }


def _app_like_chunk_text(text: str) -> str:
    """Mirror app Evidence Used behavior: render raw source text with strip only."""
    return str(text or "").strip()


def _compact_preview_text(text: str, *, limit: int = 260) -> str:
    """Compact preview mode matching publishing observability preview style."""
    preview = " ".join(str(text or "").split())
    if len(preview) > limit:
        preview = preview[:limit].rstrip() + "..."
    return preview


def _denoised_preview_text(text: str, *, limit: int = 260) -> str:
    """Compact preview plus light OCR denoising for manual labeling readability."""
    preview = _compact_preview_text(text, limit=10_000)
    preview = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", preview)
    preview = re.sub(r"([a-z])([A-Z])", r"\1 \2", preview)
    preview = re.sub(r"([A-Za-z])(\d)", r"\1 \2", preview)
    preview = re.sub(r"(\d)([A-Za-z])", r"\1 \2", preview)
    preview = re.sub(r"[ ]{2,}", " ", preview).strip()
    if len(preview) > limit:
        preview = preview[:limit].rstrip() + "..."
    return preview


def _render_markdown(
    bundle_rows: list[dict[str, Any]],
    *,
    doc_id: str,
    text_mode: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# Retrieval Review Set ({doc_id})")
    lines.append("")
    lines.append("Labeling flow per question:")
    lines.append("1. Read question + expected answer.")
    lines.append("2. Read candidate chunks (mapped by chunk_id).")
    lines.append("3. Fill `relevant_chunk_ids: []` with the supporting chunk IDs.")
    lines.append("")
    for index, row in enumerate(bundle_rows, start=1):
        lines.append(f"## Q{index}. {row['question']}")
        lines.append("")
        lines.append(f"Expected answer: {row.get('expected_answer') or '(empty)'}")
        lines.append("")
        lines.append(f"Retrieved chunk ids: [{', '.join(row.get('retrieved_chunk_ids', []))}]")
        lines.append("")

        candidates = row.get("candidate_chunks", [])
        if not candidates:
            lines.append("_No candidate chunk text found for this row._")
            lines.append("")
            lines.append("relevant_chunk_ids: []")
            lines.append("")
            continue

        lines.append("Chunk text mapping:")
        lines.append("")
        for candidate_index, chunk in enumerate(candidates, start=1):
            lines.append(
                f"{candidate_index}. chunk_id={chunk.get('chunk_id')} | page={chunk.get('page')} | citation={chunk.get('citation') or 'n/a'}"
            )
            lines.append("")
            chunk_text = str(chunk.get("text") or "")
            if text_mode == "denoised_preview":
                rendered = _denoised_preview_text(chunk_text)
            elif text_mode == "compact_preview":
                rendered = _compact_preview_text(chunk_text)
            else:
                rendered = _app_like_chunk_text(chunk_text)
            lines.append(rendered)
            lines.append("")
        lines.append("relevant_chunk_ids: []")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown/json review set from eval rows.")
    parser.add_argument("--eval-file", default="evals/publishing_eval_sample.jsonl")
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--user-id", default=None)
    parser.add_argument(
        "--catalog-file",
        default=None,
        help=(
            "Optional local chunk catalog JSON file (from prepare_retrieval_labels.py). "
            "When provided, no live workspace lookup is required."
        ),
    )
    parser.add_argument(
        "--catalog-out",
        default=None,
        help=(
            "Optional path to write/refresh chunk catalog when live workspace lookup succeeds. "
            "Defaults to evals/chunk_catalog_<doc_id>.json."
        ),
    )
    parser.add_argument("--out-md", default=None, help="Output markdown path.")
    parser.add_argument("--out-json", default=None, help="Output JSON path.")
    parser.add_argument(
        "--text-mode",
        choices=["app_like", "compact_preview", "denoised_preview"],
        default="compact_preview",
        help=(
            "How to render chunk text in markdown: "
            "app_like mirrors UI Evidence Used raw text; "
            "compact_preview uses short normalized previews; "
            "denoised_preview applies light OCR spacing heuristics."
        ),
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Number of candidate chunks to include per question in the review set.",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        raise SystemExit(f"Eval file not found: {eval_path}")

    workspace_title: str | None = None
    chunk_lookup: dict[str, Any] = {}
    default_catalog_path = Path(args.catalog_out) if args.catalog_out else Path(f"evals/chunk_catalog_{args.doc_id}.json")
    explicit_catalog_path = Path(args.catalog_file) if args.catalog_file else None

    if explicit_catalog_path and explicit_catalog_path.exists():
        catalog_chunks, workspace_title = _load_chunks_from_catalog(explicit_catalog_path)
        chunk_lookup = {item["chunk_id"]: item for item in catalog_chunks}
        if not chunk_lookup:
            raise SystemExit(f"Catalog file has no usable chunks: {explicit_catalog_path}")
    elif explicit_catalog_path and not explicit_catalog_path.exists():
        raise SystemExit(f"Catalog file not found: {explicit_catalog_path}")
    else:
        normalized_user_id = normalize_user_id(args.user_id)
        try:
            workspace = load_workspace(args.doc_id, normalized_user_id)
            chunks = workspace_chunks(workspace)
            if not chunks:
                raise SystemExit(f"Document '{args.doc_id}' has no indexed chunks.")
            workspace_title = str(workspace.get("document_title") or "").strip() or None
            chunk_lookup = {str(chunk.chunk_id): chunk for chunk in chunks}

            # Auto-refresh catalog to reduce multi-step workflows.
            catalog_payload = _catalog_payload_from_workspace(doc_id=args.doc_id, workspace=workspace, chunks=chunks)
            default_catalog_path.parent.mkdir(parents=True, exist_ok=True)
            default_catalog_path.write_text(
                json.dumps(catalog_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Chunk catalog written: {default_catalog_path}")
        except Exception as exc:
            if default_catalog_path.exists():
                print(
                    "Live workspace lookup failed; using existing local catalog instead: "
                    f"{default_catalog_path} ({type(exc).__name__})"
                )
                catalog_chunks, workspace_title = _load_chunks_from_catalog(default_catalog_path)
                chunk_lookup = {item["chunk_id"]: item for item in catalog_chunks}
                if not chunk_lookup:
                    raise SystemExit(f"Catalog file has no usable chunks: {default_catalog_path}")
            else:
                raise SystemExit(
                    "Live workspace lookup failed and no local catalog exists. "
                    f"Create one first with prepare_retrieval_labels.py or pass --catalog-file. "
                    f"Underlying error: {type(exc).__name__}: {exc}"
                ) from exc

    rows = _load_rows(eval_path)
    filtered_rows = [row for row in rows if str(row.get("doc_id") or "").strip() == args.doc_id]
    if not filtered_rows:
        raise SystemExit(f"No eval rows found for doc_id={args.doc_id}.")

    bundle_rows: list[dict[str, Any]] = []
    all_chunk_records: list[dict[str, Any]] = []
    for chunk_id, chunk in chunk_lookup.items():
        if isinstance(chunk, dict):
            all_chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "page": chunk.get("page"),
                    "chapter": chunk.get("chapter"),
                    "topic": chunk.get("topic"),
                    "citation": chunk.get("citation"),
                    "text": chunk.get("text") or "",
                }
            )
        else:
            all_chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "page": chunk.page,
                    "chapter": chunk.chapter,
                    "topic": chunk.topic,
                    "citation": chunk.citation,
                    "text": chunk.text,
                }
            )

    for row in filtered_rows:
        question = str(row.get("question") or "").strip()
        expected_answer = str(row.get("expected_answer") or "").strip()
        retrieved_ids = [str(item) for item in row.get("retrieved_chunk_ids", []) if str(item).strip()]
        retrieval_warning = row.get("retrieval_warning")
        retrieval_error = row.get("retrieval_error")

        use_lexical_candidates = bool(retrieval_warning) or bool(retrieval_error) or len(retrieved_ids) <= 2
        candidate_ids = retrieved_ids
        if use_lexical_candidates and all_chunk_records:
            query_text = f"{question} {expected_answer}".strip()
            candidate_ids = _rank_chunk_ids_by_lexical_overlap(
                chunks=all_chunk_records,
                query_text=query_text,
                limit=max(1, int(args.candidate_k)),
            )

        candidate_chunks: list[dict[str, Any]] = []
        for chunk_id in candidate_ids:
            chunk = chunk_lookup.get(chunk_id)
            if chunk is None:
                continue
            if isinstance(chunk, dict):
                candidate_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": chunk.get("page"),
                        "chapter": chunk.get("chapter"),
                        "topic": chunk.get("topic"),
                        "citation": chunk.get("citation"),
                        "text": chunk.get("text") or "",
                    }
                )
                continue
            candidate_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": chunk.page,
                    "chapter": chunk.chapter,
                    "topic": chunk.topic,
                    "citation": chunk.citation,
                    "text": chunk.text,
                }
            )
        bundle_rows.append(
            {
                "doc_id": args.doc_id,
                "question": question,
                "expected_answer": expected_answer,
                "retrieved_chunk_ids": retrieved_ids,
                "relevant_chunk_ids": [
                    str(item) for item in row.get("relevant_chunk_ids", []) if str(item).strip()
                ],
                "retrieval_warning": retrieval_warning,
                "retrieval_error": retrieval_error,
                "candidate_source": "lexical_fallback" if use_lexical_candidates else "retrieved_ids",
                "candidate_chunks": candidate_chunks,
            }
        )

    out_md = Path(args.out_md) if args.out_md else Path(f"evals/review_bundle_{args.doc_id}.md")
    out_json = Path(args.out_json) if args.out_json else Path(f"evals/review_bundle_{args.doc_id}.json")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    markdown_payload = _render_markdown(
        bundle_rows,
        doc_id=args.doc_id,
        text_mode=args.text_mode,
    )
    out_md.write_text(markdown_payload, encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "doc_id": args.doc_id,
                "document_title": workspace_title,
                "question_count": len(bundle_rows),
                "rows": bundle_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Review markdown: {out_md}")
    print(f"Review json: {out_json}")
    print(f"Questions in review set: {len(bundle_rows)}")


if __name__ == "__main__":
    main()
