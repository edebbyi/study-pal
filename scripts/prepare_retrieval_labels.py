"""prepare_retrieval_labels.py: Build label-ready retrieval records for eval curation.

Usage:
  python scripts/prepare_retrieval_labels.py \
    --doc-id <doc_id> \
    --question "What is this book a companion publication to?" \
    --k 10 \
    --catalog-out /tmp/doc_chunks.json \
    --row-out /tmp/eval_row.jsonl

This script:
1) export all chunks for a doc (for manual review),
2) retrieve top-k candidate chunks for a question,
3) produce a JSONL row template where `relevant_chunk_ids` can be filled during review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit(
        "StudyPal scripts require Python 3.11+. "
        "Run with the project environment, for example: "
        "`myenv/bin/python scripts/prepare_retrieval_labels.py ...`"
    )

# Ensure repo root is importable when running as a standalone script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.document_workspace import (  # noqa: E402
    load_workspace,
    normalize_user_id,
    retrieve_workspace_context,
    workspace_chunks,
)


def _preview(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare retrieval-labeling artifacts for one doc + question.")
    parser.add_argument("--doc-id", required=True, help="Document id from the workspace/library.")
    parser.add_argument("--question", required=True, help="Eval question text.")
    parser.add_argument("--user-id", default=None, help="Optional user id for scoped workspace retrieval.")
    parser.add_argument("--k", type=int, default=10, help="Top-k retrieval candidates to inspect.")
    parser.add_argument(
        "--catalog-out",
        default="evals/chunk_catalog.json",
        help="Path to write full chunk catalog JSON.",
    )
    parser.add_argument(
        "--row-out",
        default="evals/retrieval_label_row.jsonl",
        help="Path to append one label template row (JSONL).",
    )
    args = parser.parse_args()

    normalized_user_id = normalize_user_id(args.user_id)
    workspace = load_workspace(args.doc_id, normalized_user_id)
    all_chunks = workspace_chunks(workspace)
    if not all_chunks:
        raise SystemExit(f"Document '{args.doc_id}' has no indexed chunks.")

    catalog_path = Path(args.catalog_out)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_payload = {
        "doc_id": args.doc_id,
        "document_title": str(workspace.get("document_title") or "").strip() or None,
        "filename": str(workspace.get("filename") or "").strip() or None,
        "chunk_count": len(all_chunks),
        "chunks": [
            {
                "chunk_id": str(chunk.chunk_id),
                "page": chunk.page,
                "chapter": chunk.chapter,
                "topic": chunk.topic,
                "citation": chunk.citation,
                "preview": _preview(chunk.text),
                "text": chunk.text,
            }
            for chunk in all_chunks
        ],
    }
    catalog_path.write_text(json.dumps(catalog_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    retrieved = retrieve_workspace_context(
        workspace=workspace,
        question=args.question,
        user_id=normalized_user_id,
        top_k=max(1, int(args.k)),
    )
    candidates = [
        {
            "chunk_id": str(chunk.chunk_id),
            "score": float(chunk.score),
            "citation": chunk.citation,
            "preview": _preview(chunk.text),
        }
        for chunk in retrieved
    ]

    row_template = {
        "doc_id": args.doc_id,
        "question": args.question,
        "expected_answer": "",
        "relevant_chunk_ids": [],
        "retrieved_chunk_ids": [item["chunk_id"] for item in candidates],
        "task_type": "ask_the_book",
        "notes": "Fill relevant_chunk_ids after human review.",
    }

    row_path = Path(args.row_out)
    row_path.parent.mkdir(parents=True, exist_ok=True)
    with row_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row_template, ensure_ascii=False) + "\n")

    print(f"Chunk catalog written: {catalog_path}")
    print(f"Label template appended: {row_path}")
    print("\nTop-k retrieval candidates:")
    for idx, item in enumerate(candidates, start=1):
        print(
            f"{idx:>2}. chunk_id={item['chunk_id']} score={item['score']:.3f} "
            f"citation={item['citation'] or 'n/a'}\n    {item['preview']}"
        )


if __name__ == "__main__":
    main()
