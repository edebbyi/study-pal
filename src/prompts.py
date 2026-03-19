from __future__ import annotations


def build_answer_prompt(context: str, question: str) -> str:
    return (
        "You are Study Pal, a grounded tutor.\n"
        "Answer only from the provided notes context.\n"
        "If the answer is not supported by the context, say that clearly.\n"
        "Use a concise, student-friendly explanation.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )


def build_document_metadata_prompt(filename: str, context: str) -> str:
    return (
        "You are extracting structured document metadata from uploaded class material.\n"
        "Return valid JSON with this shape:\n"
        '{'
        '"document_title":"string",'
        '"document_topic":"string",'
        '"document_summary":"string"'
        '}\n'
        "Rules:\n"
        "- Infer a concise title that fits the material.\n"
        "- Infer a high-level topic for the whole document.\n"
        "- Write a short 1-2 sentence summary.\n"
        "- Use only information supported by the document text.\n\n"
        f"Filename:\n{filename}\n\n"
        f"Document excerpt:\n{context}"
    )


def build_quiz_prompt(
    topic: str,
    context: str,
    num_questions: int,
    weak_concepts: list[str] | None = None,
) -> str:
    reinforcement_clause = ""
    if weak_concepts:
        reinforcement_clause = (
            "\nPrioritize reinforcement for these weak concepts: "
            + ", ".join(weak_concepts)
            + "."
        )

    return (
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


def build_reteach_prompt(topic: str, weak_concepts: list[str], context: str) -> str:
    return (
        "You are reteaching a student using only the provided notes.\n"
        "Explain the weak concepts clearly and briefly, with a focus on helping them pass the next quiz.\n"
        "Do not introduce information that is not in the notes.\n\n"
        f"Topic:\n{topic}\n\n"
        f"Weak concepts:\n{', '.join(weak_concepts)}\n\n"
        f"Context:\n{context}"
    )


def build_study_plan_prompt(
    topic: str,
    weak_concepts: list[str],
    reviewed_concepts: list[str],
    context: str,
) -> str:
    weak_concepts_text = ", ".join(weak_concepts) if weak_concepts else "none"
    reviewed_concepts_text = ", ".join(reviewed_concepts) if reviewed_concepts else topic
    return (
        "You are creating a concise study plan grounded only in the provided notes.\n"
        "Return valid JSON with this shape:\n"
        '{'
        '"topic":"string",'
        '"reviewed_topics":["string"],'
        '"weak_areas":["string"],'
        '"recommended_order":["string"],'
        '"suggested_next_steps":["string","string","string"]'
        '}\n'
        "Rules:\n"
        "- Keep the plan practical and student-friendly.\n"
        "- Use only concepts supported by the notes context.\n"
        "- Stay tightly focused on the reviewed concepts instead of expanding to loosely related topics.\n"
        "- If there are weak areas, prioritize them in the recommended order.\n\n"
        f"Topic:\n{topic}\n\n"
        f"Reviewed concepts:\n{reviewed_concepts_text}\n\n"
        f"Weak concepts:\n{weak_concepts_text}\n\n"
        f"Context:\n{context}"
    )
