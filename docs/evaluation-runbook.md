# Evaluation Runbook

This guide covers offline retrieval evaluation, label prep, candidate backfilling, and human review set generation.

## Metric Framework

Live-run metrics (UI/API):

- `retrieved_chunk_count`
- `avg_relevance_score`
- `top_relevance_score`
- `context_coverage_score`
- `context_coverage_label`
- latency and review flags

Offline labeled metrics (ground truth required):

- `Hit@1`
- `Precision@k`
- `Recall@k`
- `MRR`

## 1) Run Baseline Retrieval Evaluation

```bash
myenv/bin/python scripts/evaluate_retrieval.py --eval-file evals/publishing_eval_sample.jsonl --k 5
```

Notes:

- `evals/publishing_eval_sample.jsonl` can include `retrieved_chunk_ids` for baseline/offline scoring.
- Omit `retrieved_chunk_ids` to force live retrieval against indexed docs.

## 2) Prepare Labeling Rows (Question + Candidate Chunks)

```bash
myenv/bin/python scripts/prepare_retrieval_labels.py \
  --doc-id <doc_id> \
  --question "What is this book a companion publication to?" \
  --k 10 \
  --catalog-out evals/chunk_catalog.json \
  --row-out evals/retrieval_label_rows.jsonl
```

Then manually fill `relevant_chunk_ids` in generated rows.

## 3) Backfill Retrieved Chunk IDs

```bash
myenv/bin/python scripts/fill_retrieved_chunk_ids.py \
  --eval-file evals/publishing_eval_sample.jsonl \
  --doc-id c385eadd61 \
  --user-id debroah26@gmail.com \
  --k 10
```

Options:

- `--force` overwrites existing retrieved IDs
- `--out-file <path>` writes to a separate dataset file

## 4) Build Human Review Set (Markdown + JSON)

```bash
myenv/bin/python scripts/build_eval_review_bundle.py \
  --eval-file evals/publishing_eval_sample.jsonl \
  --doc-id c385eadd61 \
  --out-md evals/review_bundle_c385eadd61.md \
  --out-json evals/review_bundle_c385eadd61.json
```

This refreshes:

- `evals/chunk_catalog_c385eadd61.json`
- markdown/json review outputs

## 5) Build Review Set Offline From Existing Catalog

```bash
myenv/bin/python scripts/build_eval_review_bundle.py \
  --eval-file evals/publishing_eval_sample.jsonl \
  --doc-id c385eadd61 \
  --catalog-file evals/chunk_catalog_c385eadd61.json \
  --out-md evals/review_bundle_c385eadd61.md \
  --out-json evals/review_bundle_c385eadd61.json
```
