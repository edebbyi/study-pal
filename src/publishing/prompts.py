"""prompts.py: Prompt builders for Publishing Mode generation tasks."""

from __future__ import annotations

from src.core.models import RetrievedChunk


def _build_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    """Build a compact citation-anchored context block for prompting."""
    sections: list[str] = []
    for chunk in retrieved_chunks:
        chunk_text = chunk.text.strip()
        if len(chunk_text) > 900:
            chunk_text = chunk_text[:900].rstrip() + "..."
        sections.append(f"[{chunk.citation}] (score={chunk.score:.3f})\n{chunk_text}")
    return "\n\n".join(sections)


def build_book_brief_prompt(
    *,
    retrieved_chunks: list[RetrievedChunk],
    audience: str | None,
    spoiler_level: str | None,
    notes: str | None,
    document_title_hint: str | None,
) -> str:
    """Build a grounded prompt for structured Book Brief generation."""
    audience_text = audience.strip() if isinstance(audience, str) and audience.strip() else "not specified"
    spoiler_text = (
        spoiler_level.strip()
        if isinstance(spoiler_level, str) and spoiler_level.strip()
        else "low (avoid major spoilers unless explicitly supported)"
    )
    notes_text = notes.strip() if isinstance(notes, str) and notes.strip() else "none"
    title_hint_text = (
        document_title_hint.strip()
        if isinstance(document_title_hint, str) and document_title_hint.strip()
        else "none"
    )
    context = _build_context(retrieved_chunks)

    return (
        "You are a publishing strategy assistant for editorial, marketing, and sales teams.\n"
        "Use only the provided grounded source excerpts.\n"
        "Do NOT invent plot details, claims, or titles beyond evidence.\n"
        "Your job is strategic synthesis, not a table-of-contents dump.\n"
        "When evidence is partial, infer cautiously from repeated patterns, chapter cues, and wording in excerpts.\n"
        "Use the exact phrase \"Insufficient evidence from provided sources.\" only when no grounded inference is possible.\n"
        "Avoid repeating the same sentence across multiple fields.\n"
        "If the document appears informational/nonfiction, infer practical themes and positioning language from headings and concepts.\n"
        "Do NOT generate channel-ready marketing copy or back-cover prose in this response.\n"
        "Respect spoiler guidance.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "title": "string|null",\n'
        '  "genre": "string|null",\n'
        '  "primary_audience": "string",\n'
        '  "secondary_audience": "string|null",\n'
        '  "reader_buyer_persona": {\n'
        '    "persona_name": "string",\n'
        '    "role_context": "string",\n'
        '    "motivation": "string",\n'
        '    "needs": "string",\n'
        '    "likely_objections": "string",\n'
        '    "discovery_channels": "string",\n'
        '    "messaging_notes": "string"\n'
        '  },\n'
        '  "core_themes": ["string"],\n'
        '  "audience_keywords": ["string"],\n'
        '  "one_sentence_positioning": "string",\n'
        '  "positioning_recommendation": "string",\n'
        '  "marketing_angles": ["string"],\n'
        '  "sales_use_case": "string",\n'
        '  "risk_flags": ["string"]\n'
        "}\n\n"
        "Output quality rules:\n"
        "- core_themes: return 3-5 concise themes when evidence exists.\n"
        "- audience_keywords: return 4-8 practical keywords when evidence exists.\n"
        "- marketing_angles: return 3-5 concrete angles when evidence exists.\n"
        "- reader_buyer_persona: concise strategic snapshot, not fictional storytelling and not channel copy.\n"
        "- reader_buyer_persona fields must be conservative when inferred; avoid unsupported specifics.\n"
        "- reader_buyer_persona.motivation: avoid claiming formal standards alignment unless explicitly supported; prefer wording like classroom instruction and curriculum planning support.\n"
        "- reader_buyer_persona.messaging_notes: avoid implying formal objectives unless explicit; prefer wording like learning needs or learning contexts.\n"
        "- one_sentence_positioning: exactly one sentence.\n"
        "- positioning_recommendation: 1-2 short internal-facing sentences (strategy guidance).\n"
        "- sales_use_case: keep concise and polished (prefer 1-2 sentences; max 3).\n"
        "- sales_use_case: prefer cautious wording such as may, can support, best suited for, or potential users include.\n"
        "- sales_use_case: avoid specific standards, certifications, adoptions, or compliance claims unless explicitly supported in sources.\n"
        "- risk_flags: 2-4 practical caveats or missing-context notes when applicable.\n\n"
        f"Audience preference: {audience_text}\n"
        f"Spoiler level: {spoiler_text}\n"
        f"Additional notes: {notes_text}\n"
        f"Document title hint: {title_hint_text}\n\n"
        f"Grounded source excerpts:\n{context}"
    )


def build_marketing_copy_prompt(
    *,
    retrieved_chunks: list[RetrievedChunk],
    output_type: str,
    tone: str | None,
    audience: str | None,
    spoiler_level: str | None,
    length: str | None,
    notes: str | None,
    document_title_hint: str | None,
) -> str:
    """Build a grounded prompt for structured marketing copy generation."""
    tone_text = tone.strip() if isinstance(tone, str) and tone.strip() else "professional and engaging"
    audience_text = audience.strip() if isinstance(audience, str) and audience.strip() else "general readers"
    spoiler_text = (
        spoiler_level.strip()
        if isinstance(spoiler_level, str) and spoiler_level.strip()
        else "low (avoid major spoilers unless clearly supported)"
    )
    length_text = length.strip() if isinstance(length, str) and length.strip() else "concise"
    notes_text = notes.strip() if isinstance(notes, str) and notes.strip() else "none"
    title_hint_text = (
        document_title_hint.strip()
        if isinstance(document_title_hint, str) and document_title_hint.strip()
        else "none"
    )
    context = _build_context(retrieved_chunks)

    return (
        "You are a publishing marketing assistant.\n"
        "Generate grounded marketing assets using only the provided source excerpts.\n"
        "Never invent unsupported plot details, outcomes, or claims.\n"
        "Respect spoiler guidance and audience fit.\n"
        "If evidence is insufficient, state only what is supported and keep uncertainty explicit.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "output_type": "string",\n'
        '  "copy": "string",\n'
        '  "rationale": "string"\n'
        "}\n\n"
        "Output type guidance:\n"
        "- back_cover: compelling back-cover copy\n"
        "- newsletter: email blurb with clear hook\n"
        "- bookstore_pitch: short bookseller-facing sales pitch\n"
        "- instagram_caption: social caption with concise energy\n"
        "- tiktok_hooks: hook-forward lines for short video\n"
        "- author_interview_questions: interview prompts grounded in themes\n"
        "- book_club_questions: discussion prompts for readers\n\n"
        "Output quality rules:\n"
        "- Rationale must be 1-2 concise sentences.\n"
        "- Rationale should summarize grounding and safety (no unsupported claims), not a long page-by-page walkthrough.\n\n"
        f"Requested output_type: {output_type}\n"
        f"Tone: {tone_text}\n"
        f"Audience: {audience_text}\n"
        f"Spoiler level: {spoiler_text}\n"
        f"Length guidance: {length_text}\n"
        f"Additional notes: {notes_text}\n"
        f"Document title hint: {title_hint_text}\n\n"
        f"Grounded source excerpts:\n{context}"
    )
