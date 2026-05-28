# API Reference

Base URL: `http://localhost:8000`

## Start API

```bash
make api
```

## Health

```bash
curl http://localhost:8000/api/health
```

Expected response:

```json
{"status":"ok","service":"studypal-api","version":"1.1.0"}
```

## Route Groups

- `/api/documents`
- `/api/ask`
- `/api/publishing`
- `/api/evaluation`
- `/api/observability`

## Document-grounded Q&A

```bash
curl -X POST http://localhost:8000/api/documents/<doc_id>/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?"}'
```

## Publishing Book Brief

```bash
curl -X POST http://localhost:8000/api/publishing/<doc_id>/book-brief \
  -H "Content-Type: application/json" \
  -d '{"audience":"Book club readers","spoiler_level":"low","notes":"Highlight marketing hooks."}'
```

## Publishing Marketing Copy

```bash
curl -X POST http://localhost:8000/api/publishing/<doc_id>/marketing-copy \
  -H "Content-Type: application/json" \
  -d '{"output_type":"back_cover","tone":"cinematic","audience":"adult fiction readers","spoiler_level":"low"}'
```

## Rate A Run

```bash
curl -X POST http://localhost:8000/api/runs/<run_id>/rate \
  -H "Content-Type: application/json" \
  -d '{"user_rating":5,"user_feedback":"Grounded and on-brand."}'
```

## List Runs For One Document

```bash
curl http://localhost:8000/api/documents/<doc_id>/runs
```

## Observability Health

```bash
curl http://localhost:8000/api/observability/health
```
