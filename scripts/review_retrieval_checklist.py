"""review_retrieval_checklist.py: Streamlit checklist for retrieval relevance labeling.

Run:
  python -m streamlit run scripts/review_retrieval_checklist.py -- \
    --bundle-file evals/review_bundle_c385eadd61.json \
    --eval-file evals/publishing_eval_sample.jsonl

What it does:
1) renders each question with candidate chunk checkboxes,
2) autosaves checked chunk ids to a local state file,
3) captures optional reviewer feedback notes,
4) can write labels/notes back into:
   - the review bundle JSON (`rows[].relevant_chunk_ids`), and/or
   - eval JSONL rows (`relevant_chunk_ids`).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bundle-file", default="evals/review_bundle_c385eadd61.json")
    parser.add_argument("--eval-file", default="evals/publishing_eval_sample.jsonl")
    parser.add_argument("--state-file", default=None)
    args, _unknown = parser.parse_known_args()
    return args


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Line {line_number} in {path} is not a JSON object.")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row_key(index: int, question: str) -> str:
    return f"{index}::{question.strip()}"


def _default_state_path(bundle_path: Path, doc_id: str) -> Path:
    return bundle_path.with_name(f"review_labels_{doc_id}.json")


def _load_review_state(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    if not path.exists():
        return {}, {}
    payload = _load_json(path)
    labels = payload.get("labels")
    feedback = payload.get("feedback")
    if not isinstance(labels, dict):
        labels = {}
    if not isinstance(feedback, dict):
        feedback = {}

    normalized: dict[str, list[str]] = {}
    for key, values in labels.items():
        if not isinstance(values, list):
            continue
        normalized[str(key)] = [str(item) for item in values if str(item).strip()]

    normalized_feedback: dict[str, str] = {}
    for key, value in feedback.items():
        text = str(value or "").strip()
        if text:
            normalized_feedback[str(key)] = text
    return normalized, normalized_feedback


def _save_review_state(
    *,
    path: Path,
    doc_id: str,
    bundle_path: Path,
    labels: dict[str, list[str]],
    feedback: dict[str, str],
    current_index: int,
) -> None:
    payload: dict[str, Any] = {
        "doc_id": doc_id,
        "bundle_file": str(bundle_path),
        "current_index": int(current_index),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "feedback": feedback,
    }
    _write_json(path, payload)


def _normalize_chunk_text(text: str, limit: int = 380) -> str:
    preview = _denoise_chunk_text(str(text or ""))
    if len(preview) > limit:
        return preview[:limit].rstrip() + "..."
    return preview


def _denoise_chunk_text(text: str) -> str:
    # Flatten whitespace first.
    cleaned = " ".join(text.split())
    # Join long runs of single-letter tokens like "T h e W i z a r d".
    cleaned = re.sub(
        r"(?:\b[A-Za-z]\b\s+){4,}\b[A-Za-z]\b",
        lambda m: "".join(re.findall(r"[A-Za-z]", m.group(0))),
        cleaned,
    )
    # Light OCR spacing repair around punctuation/case changes.
    cleaned = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([A-Za-z])(\d)", r"\1 \2", cleaned)
    cleaned = re.sub(r"(\d)([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _retrieval_status_line(
    *,
    candidate_chunks: list[dict[str, Any]] | Any,
    warning: str,
    error: str,
    candidate_source: str,
) -> str:
    """Build a concise, reviewer-friendly retrieval status line."""
    has_candidates = isinstance(candidate_chunks, list) and bool(candidate_chunks)
    fallback_signals = "fallback_used" in warning or candidate_source == "lexical_fallback"
    if not has_candidates:
        return "No candidates available for this question."
    if fallback_signals:
        return "No grounded hit found; showing fallback candidates for manual labeling."
    if error:
        return "Retriever reported an error; showing available candidates."
    return "Grounded retrieval candidates available."


def _apply_labels_to_bundle(
    *,
    bundle_payload: dict[str, Any],
    labels: dict[str, list[str]],
    feedback: dict[str, str],
) -> tuple[dict[str, Any], int]:
    rows = bundle_payload.get("rows")
    if not isinstance(rows, list):
        return bundle_payload, 0

    changed = 0
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question = str(row.get("question") or "")
        key = _row_key(row_index, question)
        selected = labels.get(key, [])
        note = feedback.get(key, "")
        existing = [str(item) for item in row.get("relevant_chunk_ids", []) if str(item).strip()]
        if existing != selected:
            row["relevant_chunk_ids"] = selected
            changed += 1
        row["found_relevance"] = bool(selected)
        row["reviewer_feedback"] = note
        row.pop("label_status", None)
    return bundle_payload, changed


def _apply_labels_to_eval_jsonl(
    *,
    eval_rows: list[dict[str, Any]],
    bundle_rows: list[dict[str, Any]],
    labels: dict[str, list[str]],
    feedback: dict[str, str],
    doc_id: str,
) -> int:
    mapping: dict[str, dict[str, Any]] = {}
    for row_index, bundle_row in enumerate(bundle_rows):
        if not isinstance(bundle_row, dict):
            continue
        question = str(bundle_row.get("question") or "").strip()
        if not question:
            continue
        key = _row_key(row_index, question)
        mapping[question] = {
            "selected": labels.get(key, []),
            "feedback": feedback.get(key, ""),
        }

    changed = 0
    for row in eval_rows:
        if str(row.get("doc_id") or "").strip() != doc_id:
            continue
        question = str(row.get("question") or "").strip()
        if question not in mapping:
            continue
        selected = mapping[question]["selected"]
        note = str(mapping[question]["feedback"] or "").strip()
        existing = [str(item) for item in row.get("relevant_chunk_ids", []) if str(item).strip()]
        if existing != selected:
            row["relevant_chunk_ids"] = selected
            changed += 1
        row["found_relevance"] = bool(selected)
        row["reviewer_feedback"] = note
        row.pop("label_status", None)
    return changed


def main() -> None:
    args = _parse_args()
    bundle_path = Path(args.bundle_file)
    eval_path = Path(args.eval_file)

    st.set_page_config(page_title="Retrieval Checklist Review", layout="wide")
    st.title("Retrieval Checklist Review")

    if not bundle_path.exists():
        st.error(f"Bundle file not found: {bundle_path}")
        st.stop()

    bundle_payload = _load_json(bundle_path)
    doc_id = str(bundle_payload.get("doc_id") or "").strip()
    rows = bundle_payload.get("rows")
    if not doc_id or not isinstance(rows, list) or not rows:
        st.error("Bundle JSON is missing a usable `doc_id` or `rows`.")
        st.stop()

    state_path = Path(args.state_file) if args.state_file else _default_state_path(bundle_path, doc_id)
    saved_labels, saved_feedback = _load_review_state(state_path)

    if "review_index" not in st.session_state:
        st.session_state.review_index = 0
    review_index = int(st.session_state.review_index)
    review_index = max(0, min(review_index, len(rows) - 1))
    st.session_state.review_index = review_index
    last_rendered_index = st.session_state.get("_last_rendered_review_index")
    row_changed = last_rendered_index != review_index
    st.session_state["_last_rendered_review_index"] = review_index

    all_labels: dict[str, list[str]] = {}
    all_feedback: dict[str, str] = {}
    total_selected = 0

    st.sidebar.subheader("Files")
    st.sidebar.text(f"Bundle: {bundle_path}")
    st.sidebar.text(f"Eval JSONL: {eval_path}")
    st.sidebar.text(f"Label state: {state_path}")
    show_debug_retrieval = st.sidebar.checkbox("Show retrieval debug details", value=False)
    st.sidebar.divider()

    if st.sidebar.button("Prev Question", use_container_width=True):
        st.session_state.review_index = max(0, review_index - 1)
        st.rerun()
    if st.sidebar.button("Next Question", use_container_width=True):
        st.session_state.review_index = min(len(rows) - 1, review_index + 1)
        st.rerun()

    chosen = st.sidebar.number_input(
        "Jump to question #",
        min_value=1,
        max_value=len(rows),
        value=review_index + 1,
        step=1,
    )
    if int(chosen) - 1 != review_index:
        st.session_state.review_index = int(chosen) - 1
        st.rerun()

    for row_index, row_obj in enumerate(rows):
        if not isinstance(row_obj, dict):
            continue
        question = str(row_obj.get("question") or "")
        key = _row_key(row_index, question)
        default_selected = saved_labels.get(
            key,
            [str(item) for item in row_obj.get("relevant_chunk_ids", []) if str(item).strip()],
        )
        candidate_chunks = row_obj.get("candidate_chunks", [])
        selected_for_row: list[str] = list(default_selected)
        # Important: only derive labels from live checkboxes on the currently visible row.
        # For other rows, keep persisted selections to avoid accidental overwrite.
        if row_index == review_index and isinstance(candidate_chunks, list):
            selected_for_row = []
            for chunk in candidate_chunks:
                if not isinstance(chunk, dict):
                    continue
                chunk_id = str(chunk.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue
                cb_key = f"cb::{key}::{chunk_id}"
                # When navigating between questions, rehydrate visible checkboxes from
                # persisted labels so UI stays aligned with saved backend state.
                if row_changed or cb_key not in st.session_state:
                    st.session_state[cb_key] = chunk_id in default_selected
                if st.session_state[cb_key]:
                    selected_for_row.append(chunk_id)
        all_labels[key] = selected_for_row
        total_selected += len(selected_for_row)
        feedback_key = f"fb::{key}"
        if feedback_key not in st.session_state:
            default_note = saved_feedback.get(
                key,
                str(row_obj.get("reviewer_feedback") or "").strip(),
            )
            st.session_state[feedback_key] = default_note
        all_feedback[key] = str(st.session_state.get(feedback_key, "") or "").strip()

    _save_review_state(
        path=state_path,
        doc_id=doc_id,
        bundle_path=bundle_path,
        labels=all_labels,
        feedback=all_feedback,
        current_index=review_index,
    )

    current = rows[review_index]
    question = str(current.get("question") or "")
    expected = str(current.get("expected_answer") or "")
    current_key = _row_key(review_index, question)
    candidate_chunks = current.get("candidate_chunks", [])
    retrieved_ids = [str(item) for item in current.get("retrieved_chunk_ids", []) if str(item).strip()]

    st.caption(
        f"Doc `{doc_id}` | Question {review_index + 1}/{len(rows)} | "
        f"Autosave active ({state_path.name}) | Selected chunks total: {total_selected}"
    )
    st.subheader(f"Q{review_index + 1}. {question}")
    if expected:
        st.markdown(f"**Expected answer:** {expected}")
    st.markdown(f"**Retrieved chunk ids:** `{retrieved_ids}`")

    warning = str(current.get("retrieval_warning") or "").strip()
    error = str(current.get("retrieval_error") or "").strip()
    candidate_source = str(current.get("candidate_source") or "").strip()
    current_found_relevance = bool(all_labels.get(current_key, []))
    retrieval_status = _retrieval_status_line(
        candidate_chunks=candidate_chunks,
        warning=warning,
        error=error,
        candidate_source=candidate_source,
    )
    st.markdown(f"**Retrieval status:** {retrieval_status}")
    if show_debug_retrieval and (warning or error or candidate_source):
        with st.expander("Retrieval debug details"):
            if candidate_source:
                st.text(f"candidate_source: {candidate_source}")
            if warning:
                st.text(f"retrieval_warning: {warning}")
            if error:
                st.text(f"retrieval_error: {error}")

    st.markdown("### Candidate Chunks")
    max_preview = st.slider("Preview length", min_value=180, max_value=8000, value=1200, step=60)
    show_raw = st.checkbox("Show raw OCR text", value=False)
    if not isinstance(candidate_chunks, list) or not candidate_chunks:
        st.info("No candidate chunks for this row.")
    else:
        for candidate_index, chunk in enumerate(candidate_chunks, start=1):
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            page = chunk.get("page")
            citation = str(chunk.get("citation") or "n/a")
            raw_text = str(chunk.get("text") or "")
            text = _normalize_chunk_text(raw_text, limit=max_preview)
            cb_key = f"cb::{current_key}::{chunk_id}"
            st.checkbox(
                f"[{candidate_index}] chunk_id={chunk_id} | page={page}",
                key=cb_key,
            )
            st.caption(citation)
            st.text(text)
            if show_raw:
                with st.expander(f"Raw text: chunk {chunk_id}"):
                    st.text(raw_text)

    st.markdown(f"**Current `relevant_chunk_ids`:** `{all_labels.get(current_key, [])}`")
    st.markdown(f"**Current `found_relevance`:** `{str(current_found_relevance).lower()}`")
    st.markdown("### Feedback")
    st.text_area(
        "Reviewer feedback (optional)",
        key=f"fb::{current_key}",
        placeholder="Add any quick note about retrieval quality or why chunks were/weren't selected.",
        height=90,
    )
    st.divider()
    st.markdown("### Save Labels")
    st.caption("Saves to both eval JSONL (source of truth) and review bundle JSON (readable mirror).")
    if st.button("Save labels", type="primary", use_container_width=True):
        if not eval_path.exists():
            st.error(f"Eval file not found: {eval_path}")
        else:
            eval_rows = _load_jsonl(eval_path)
            eval_changed = _apply_labels_to_eval_jsonl(
                eval_rows=eval_rows,
                bundle_rows=rows,
                labels=all_labels,
                feedback=all_feedback,
                doc_id=doc_id,
            )
            _write_jsonl(eval_path, eval_rows)
            message = f"Updated eval rows: {eval_changed}"

            refreshed_payload = _load_json(bundle_path)
            updated_payload, bundle_changed = _apply_labels_to_bundle(
                bundle_payload=refreshed_payload,
                labels=all_labels,
                feedback=all_feedback,
            )
            _write_json(bundle_path, updated_payload)
            message += f" | Updated bundle rows: {bundle_changed}"

            st.success(message)
            no_support_count = 0
            with_support_count = 0
            selected_chunk_total = 0
            for row_index, row_obj in enumerate(rows):
                if not isinstance(row_obj, dict):
                    continue
                key = _row_key(row_index, str(row_obj.get("question") or ""))
                selected = all_labels.get(key, [])
                selected_chunk_total += len(selected)
                if not selected:
                    no_support_count += 1
                else:
                    with_support_count += 1
            total_questions = with_support_count + no_support_count
            st.info(
                "Progress summary: "
                f"{with_support_count}/{total_questions} question(s) have selected relevant chunks; "
                f"{no_support_count}/{total_questions} currently have none "
                f"(includes unanswered questions). Total selected chunks: {selected_chunk_total}."
            )


if __name__ == "__main__":
    main()
