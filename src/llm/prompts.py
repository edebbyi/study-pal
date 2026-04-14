"""prompts.py: Prompt builders and Langfuse prompt resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass

from langfuse import get_client

from src.core.config import settings
from src.core.observability import configure_langfuse_environment, langfuse_enabled


@dataclass(frozen=True)
class PromptBundle:
    text: str
    prompt: object | None
    name: str



def _render_langfuse_prompt(name: str, variables: dict[str, object]) -> PromptBundle | None:
    """Try to render a prompt from Langfuse.
    
    Args:
        name (str): Input parameter.
        variables (dict[str, object]): Input parameter.
    
    Returns:
        PromptBundle | None: Result value.
    """

    if not langfuse_enabled():
        return None
    if not configure_langfuse_environment():
        return None

    try:
        langfuse = get_client()
        prompt = None
        if hasattr(langfuse, "get_prompt"):
            try:
                if settings.langfuse_prompt_version:
                    prompt = langfuse.get_prompt(name, version=settings.langfuse_prompt_version)
                else:
                    prompt = langfuse.get_prompt(name)
            except TypeError:
                prompt = langfuse.get_prompt(name)
        elif hasattr(langfuse, "prompt"):
            prompt = langfuse.prompt(name)
        if prompt is None:
            return None

        if hasattr(prompt, "compile"):
            return PromptBundle(text=prompt.compile(**variables), prompt=prompt, name=name)
        if hasattr(prompt, "render"):
            return PromptBundle(text=prompt.render(**variables), prompt=prompt, name=name)

        raw_prompt = None
        if isinstance(prompt, str):
            raw_prompt = prompt
        elif hasattr(prompt, "prompt"):
            raw_prompt = prompt.prompt
        elif hasattr(prompt, "text"):
            raw_prompt = prompt.text

        if isinstance(raw_prompt, str):
            try:
                return PromptBundle(text=raw_prompt.format(**variables), prompt=prompt, name=name)
            except (KeyError, ValueError):
                return PromptBundle(text=raw_prompt, prompt=prompt, name=name)
    except Exception:
        return None

    return None


def _resolve_prompt(
    *,
    prompt_name: str,
    variables: dict[str, object],

    fallback: str,
) -> PromptBundle:
    """Prefer Langfuse prompts when available, otherwise use a local fallback.
    
    Args:
        prompt_name (str): Input parameter.
        variables (dict[str, object]): Input parameter.
        fallback (str): Input parameter.
    
    Returns:
        PromptBundle: Result value.
    """

    rendered = _render_langfuse_prompt(prompt_name, variables)
    if rendered:
        return rendered
    return PromptBundle(text=fallback, prompt=None, name=prompt_name)  # local fallback keeps app usable offline



def build_answer_prompt(context: str, question: str) -> PromptBundle:
    """Build the prompt for note-grounded answers.
    
    Args:
        context (str): Context string for prompts or parsing.
        question (str): The user question to answer.
    
    Returns:
        PromptBundle: Result value.
    """

    fallback = (
        "You are Study Pal, a grounded tutor.\n"
        "Answer only from the provided notes context.\n"
        "If the answer is not supported by the context, say that clearly.\n"
        "Use a concise, student-friendly explanation.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )
    return _resolve_prompt(
        prompt_name=settings.langfuse_prompt_answer,
        variables={"context": context, "question": question},
        fallback=fallback,
    )


def build_structured_answer_prompt(
    *,
    context: str,
    question: str,
    persona_name: str,
    example: str,

    chat_history: str,
) -> PromptBundle:
    """Build the prompt for a structured answer plus action lanes.
    
    Args:
        context (str): Context string for prompts or parsing.
        question (str): The user question to answer.
        persona_name (str): Input parameter.
        example (str): Input parameter.
        chat_history (str): Input parameter.
    
    Returns:
        PromptBundle: Result value.
    """

    example_block = f"{example}\n\n" if example.strip() else ""  # optional few-shot example
    history_block = f"Previous conversation:\n{chat_history}\n\n" if chat_history.strip() else ""  # only include when available
    fallback = (
        "You are Study Pal, a grounded tutor.\n"
        "Your goal is to answer the user's question AND provide \"Bridges\" to keep them learning.\n\n"
        "RULES:\n"
        "1. Use ONLY the provided notes context.\n"
        "2. Always infer the best possible answer from the context when any relevant information exists.\n"
        "3. Only say \"I couldn't find that in the notes yet.\" when NO context is retrieved.\n"
        "4. Always provide a bridge button based on something that IS in the notes.\n"
        "5. Keep the answer to a MAXIMUM of 4 sentences.\n"
        f"6. Use the persona: {persona_name}.\n"
        "7. Return ONLY valid JSON. No extra commentary.\n\n"
        "Button Label Rule (Info Lane):\n"
        "- Do NOT use \"Explore [Topic]\" or \"Learn about [Topic]\".\n"
        "- Use one of these hook styles:\n"
        "  - The Mystery: \"The secret role of the [Topic]...\"\n"
        "  - The Connection: \"How [Topic] actually changes your [Result]...\"\n"
        "  - The Question: \"Why is the [Topic] so critical for [Function]?\"\n"
        "- Must start with exactly ONE relevant emoji (e.g., 🧠, ⚡, 🧩).\n\n"
        "Action Metadata Rule:\n"
        "- Identify a noun + verb pair in the notes and shape the hook around it.\n\n"
        "Return valid JSON with this shape:\n"
        "{\n"
        "  \"answer\": \"Concise, student-friendly explanation.\",\n"
        "  \"citations\": [\"Doc A, p.2\", \"Doc B, p.5\"],\n"
        "  \"topic_subject\": \"Short subject label for this quiz goal (e.g., Dendrites)\",\n"
        "  \"info_lane\": {\n"
        "    \"button_label\": \"🧠 Hook-style label here\",\n"
        "    \"query\": \"Specific follow-up query this button triggers.\"\n"
        "  },\n"
        "  \"quiz_lane\": {\n"
        "    \"button_label\": \"Test your knowledge on this\",\n"
        "    \"intent\": \"START_QUIZ_LOOP\"\n"
        "  }\n"
        "}\n\n"
        f"{example_block}"
        f"{history_block}"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )
    return _resolve_prompt(
        prompt_name=settings.langfuse_prompt_structured_answer,
        variables={
            "context": context,
            "question": question,
            "persona_name": persona_name,
            "example": example,
            "chat_history": chat_history,
        },
        fallback=fallback,
    )



def build_document_metadata_prompt(filename: str, context: str) -> PromptBundle:
    """Build the prompt for extracting document metadata.
    
    Args:
        filename (str): Filename associated with the document.
        context (str): Context string for prompts or parsing.
    
    Returns:
        PromptBundle: Result value.
    """

    fallback = (
        "You are extracting structured metadata from class material.\n\n"
        "RULES:\n"
        "1. Use ONLY the provided document excerpt.\n"
        "2. Return ONLY valid JSON. No extra commentary.\n"
        "3. Keep titles concise and professional.\n"
        "4. The key_hooks must be curiosity gaps (e.g., \"How neurons fire\") not generic nouns.\n"
        "5. Provide exactly 4 key_hooks if possible.\n\n"
        "Return valid JSON with this shape:\n"
        "{\n"
        "  \"document_title\": \"string\",\n"
        "  \"document_topic\": \"string\",\n"
        "  \"document_summary\": \"string\",\n"
        "  \"key_hooks\": [\"string\", \"string\", \"string\", \"string\"]\n"
        "}\n\n"
        f"Filename:\n{filename}\n\n"
        f"Document excerpt:\n{context}"
    )
    return _resolve_prompt(
        prompt_name=settings.langfuse_prompt_document_metadata,
        variables={"filename": filename, "context": context},
        fallback=fallback,
    )


def build_quiz_prompt(
    topic: str,
    context: str,
    num_questions: int,

    weak_concepts: list[str] | None = None,
) -> PromptBundle:
    """Build the prompt for generating a quiz.
    
    Args:
        topic (str): Topic label for the request.
        context (str): Context string for prompts or parsing.
        num_questions (int): Input parameter.
        weak_concepts (list[str] | None): Input parameter.
    
    Returns:
        PromptBundle: Result value.
    """

    reinforcement_clause = ""
    if weak_concepts:
        reinforcement_clause = (
            "\nPrioritize reinforcement for these weak concepts: "
            + ", ".join(weak_concepts)
            + "."
        )

    fallback = (
        "You are generating a multiple-choice study quiz grounded only in the provided notes.\n"
        "Return valid JSON with this shape:\n"
        '{'
        '"title": "string", '
        '"topic": "string", '
        '"questions": ['
        '{"prompt":"string","options":["string","string","string","string"],'
        '"correct_answer":"string","concept_tag":"string"}'
        "]"
        '}\n'
        "Rules:\n"
        f"- Create exactly {num_questions} questions.\n"
        "- Each question must have 4 answer options.\n"
        "- The correct_answer must exactly match one option.\n"
        "- Use only information supported by the notes context.\n"
        "- Keep wording clear and exam-oriented."
        f"{reinforcement_clause}\n\n"
        f"Topic:\n{topic}\n\n"
        f"Context:\n{context}"
    )
    return _resolve_prompt(
        prompt_name=settings.langfuse_prompt_quiz,
        variables={
            "topic": topic,
            "context": context,
            "num_questions": num_questions,
            "weak_concepts": weak_concepts or [],
        },
        fallback=fallback,
    )



def build_reteach_prompt(topic: str, weak_concepts_with_error: str, context: str) -> PromptBundle:
    """Build the prompt for reteaching weak concepts.
    
    Args:
        topic (str): Topic label for the request.
        weak_concepts_with_error (str): Input parameter.
        context (str): Context string for prompts or parsing.
    
    Returns:
        PromptBundle: Result value.
    """

    fallback = (
        "You are Study Pal. A student misunderstood a concept during their quiz.\n"
        "Your goal is to correct the mental model, not just cite the notes.\n\n"
        "RULES:\n"
        "1. Use ONLY the provided context.\n"
        "2. Address the \"Why\": Explain the logic behind the correct concept.\n"
        "3. Contrast: Explicitly mention the difference between the common misunderstanding and the reality.\n"
        "4. The mini_check: Must be a single YES/NO scenario question that tests the logic you just explained.\n"
        "5. Explain, don't just quote. Synthesize the notes into a clear \"how it works\" explanation.\n"
        "6. Do not list filenames; integrate the source into the explanation if needed.\n\n"
        "Return valid JSON with this shape:\n"
        "{\n"
        "  \"concept\": \"string\",\n"
        "  \"explanation\": \"2-3 sentences explaining the mechanism clearly.\",\n"
        "  \"contrast\": \"What the user got wrong vs. what is true.\",\n"
        "  \"mini_check\": \"A scenario-based YES/NO question.\",\n"
        "  \"mini_check_answer\": \"yes|no\"\n"
        "}\n\n"
        f"Target Concept & Error:\n{weak_concepts_with_error}\n\n"
        f"Context:\n{context}"
    )
    return _resolve_prompt(
        prompt_name=settings.langfuse_prompt_reteach,
        variables={
            "topic": topic,
            "weak_concepts_with_error": weak_concepts_with_error,
            "context": context,
        },
        fallback=fallback,
    )


def build_study_plan_prompt(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str],

    context: str,
) -> PromptBundle:
    """Build the prompt for creating a study plan.
    
    Args:
        topic (str): Topic label for the request.
        weak_concepts (list[str]): Input parameter.
        reviewed_concepts (list[str]): Input parameter.
        context (str): Context string for prompts or parsing.
    
    Returns:
        PromptBundle: Result value.
    """

    fallback = (
        "You are creating a Session Wrap-up for a student who just finished a 3-round study loop.\n\n"
        "RULES:\n"
        "1. Use ONLY the provided context and reviewed/weak concepts.\n"
        "2. Return ONLY valid JSON. No extra commentary.\n"
        "3. summary must be 2 sentences max.\n"
        "4. mastery_score should look like \"85%\".\n"
        "5. If no numeric score is provided by the app, estimate mastery_score based on how many weak concepts remain vs reviewed concepts.\n"
        "6. next_step_lane.button_label should be action-oriented and specific (no \"Learn more\").\n"
        "7. next_step_lane should point to the weakest area when possible.\n\n"
        "Return valid JSON with this shape:\n"
        "{\n"
        "  \"mastery_score\": \"string\",\n"
        "  \"summary\": \"string\",\n"
        "  \"strengths\": [\"string\"],\n"
        "  \"weak_areas\": [\"string\"],\n"
        "  \"next_step_lane\": {\n"
        "    \"button_label\": \"string\",\n"
        "    \"query\": \"string\"\n"
        "  }\n"
        "}\n\n"
        f"Reviewed concepts:\n{', '.join(reviewed_concepts) if reviewed_concepts else 'none'}\n\n"
        f"Weak concepts:\n{', '.join(weak_concepts) if weak_concepts else 'none'}\n\n"
        f"Context:\n{context}"
    )
    return _resolve_prompt(
        prompt_name=settings.langfuse_prompt_study_plan,
        variables={
            "topic": topic,
            "weak_concepts": weak_concepts,
            "reviewed_concepts": reviewed_concepts,
            "context": context,
        },
        fallback=fallback,
    )


def build_follow_up_prompt(
    question: str,
    answer: str,
    context: str,

    used_fallback: bool,
) -> PromptBundle:
    """Build a prompt for a short engagement follow-up.
    
    Args:
        question (str): The user question to answer.
        answer (str): Input parameter.
        context (str): Context string for prompts or parsing.
        used_fallback (bool): Input parameter.
    
    Returns:
        PromptBundle: Result value.
    """

    fallback = (
        "You are a study coach helping a learner stay engaged.\n"
        "Write one short follow-up question that encourages deeper learning.\n"
        "Start the question with: \"Would you like me to explain\".\n"
        "If the answer is incomplete because notes were missing, ask the user to share more notes.\n"
        "Return valid JSON with this shape:\n"
        '{"follow_up":"string","action":"quiz|example|explain|clarify"}\n\n'
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Used fallback:\n{str(used_fallback).lower()}\n\n"
        f"Context:\n{context}"
    )
    return _resolve_prompt(
        prompt_name=settings.langfuse_prompt_follow_up,
        variables={
            "question": question,
            "answer": answer,
            "context": context,
            "used_fallback": used_fallback,
        },
        fallback=fallback,
    )
