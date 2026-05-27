# Engineering Standards

This document defines the minimum quality bar for code changes in Study Pal.

## Goals

- Keep the system reliable and easy to change.
- Keep modules small, focused, and testable.
- Catch regressions before merge.

## Definition Of Done

A change is ready to merge when all are true:

- Lint passes (`make lint`).
- Tests pass (`make test`).
- New behavior has tests (unit or integration).
- README/docs are updated when user-visible behavior changes.
- No unresolved TODO/FIXME without a linked issue/ticket.

## Architecture Rules

- Keep dependency flow one-way:
  - `router -> service -> provider adapter`.
- Avoid calling provider SDKs directly from routers.
- Avoid importing UI/state modules into lower-level data or service modules.
- New modules should have one primary responsibility.

## File Size And Structure

- Prefer files under ~400 lines for core logic modules.
- If a file grows beyond this range, extract by responsibility:
  - request/response models
  - orchestration
  - provider adapters
  - parsing/formatting
  - error handling

## Typing And Data Contracts

- Use typed DTOs/models for service boundaries.
- Avoid untyped dict payloads between layers when a model is feasible.
- Avoid `Any` unless wrapping third-party SDK boundaries.

## Error Handling

- Raise typed domain errors in services.
- Map domain errors to HTTP responses in routers.
- Include actionable error messages; avoid leaking secrets or raw provider payloads.

## Logging And Observability

- Log key lifecycle events with lightweight structured metadata.
- Never log secrets, raw API keys, or sensitive user content.
- Tracing/logging failures must not break core request flow.

## Testing Strategy

- Unit tests for pure logic.
- Integration-style tests for service-layer orchestration.
- Provider/network behavior should be mocked in default test runs.
- Keep tests deterministic and isolated from external services by default.

## Documentation Standards

- Keep `README.md` concise and scannable.
- Place detailed commands/runbooks in `docs/*`.
- Add or update runbook pages when adding operational workflows.

## Commit And PR Hygiene

- Use focused commits by concern (feature, eval data, docs).
- Do not include scratch files, screenshots, or local notes unless required.
- Follow the PR checklist template and explain tradeoffs/risk clearly.
