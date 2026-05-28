# Observability Runbook

## Overview

Study Pal integrates three observability systems:

- Langfuse: prompt template versioning, traces/events, feedback scores
- Arize Phoenix: retrieval trace inspection and latency/debug analysis
- MLflow: experiment params, metrics, and artifacts

If any one is unavailable, API responses continue and only that logging layer is skipped.

## Environment Variables

### Langfuse

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_BASE_URL` (default `https://cloud.langfuse.com`)
- `LANGFUSE_PROMPT_VERSION` (optional)

### Arize Phoenix

- `PHOENIX_COLLECTOR_ENDPOINT`
- `PHOENIX_PROJECT_NAME`
- `PHOENIX_API_KEY` (optional, if your collector requires auth)

### MLflow

- `MLFLOW_TRACKING_URI` (optional; local file tracking if empty)
- `MLFLOW_EXPERIMENT_NAME`

## Publishing UI Review Signals

Publishing mode surfaces:

- grounded in source
- unsupported claims detected
- missing context present
- human review recommended
- context coverage
- model, latency, retrieved chunks, relevance scores

## Outage Behavior

When dependencies are unavailable:

- Phoenix down/not configured:
  - trace creation is skipped or downgraded
  - `phoenix_trace_id` may be null
- MLflow down/unavailable:
  - param/metric/artifact logging is skipped
  - `mlflow_run_id` may be null

## Quick Checks

```bash
curl http://127.0.0.1:8000/api/observability/health
```

## Timeout Defaults

- `API_TIMEOUT_ASK_SECONDS=30`
- `API_TIMEOUT_POSITIONING_SECONDS=45`
- `API_TIMEOUT_MARKETING_SECONDS=40`

If timeouts rise:

- reduce request scope
- reduce retrieval breadth (`top_k`) for targeted asks
- retry once after transient provider latency spikes

## Live Vs Labeled Metrics

Live run metrics:

- `retrieved_chunk_count`
- `avg_relevance_score`
- `top_relevance_score`
- `context_coverage_score`
- `context_coverage_label`
- latency and review flags

Offline labeled retrieval metrics:

- `Hit@1`
- `Precision@k`
- `Recall@k`
- `MRR`
