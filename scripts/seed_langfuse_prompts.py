"""seed_langfuse_prompts.py: Seed Langfuse prompt templates for Study Pal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langfuse import Langfuse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # Ensure local imports resolve when run as a script.

from src.core.config import settings  # noqa: E402
from src.core.observability import configure_langfuse_environment, langfuse_enabled  # noqa: E402


def _prompt_templates() -> dict[str, str]:
    """Return the default Langfuse prompt templates.

    Returns:
        dict[str, str]: Prompt name to prompt text mappings.
    """
    return {
        settings.langfuse_prompt_answer: (
            "You are Study Pal, a grounded tutor.\n"
            "Answer only from the provided notes context.\n"
            "If the answer is not supported by the context, say that clearly.\n"
            "Use a concise, student-friendly explanation.\n\n"
            "Context:\n{{context}}\n\n"
            "Question:\n{{question}}"
        ),
        settings.langfuse_prompt_structured_answer: (
            "You are Study Pal, a grounded tutor.\n"
            "Your goal is to answer the user's question AND provide \"Bridges\" to keep them learning.\n\n"
            "RULES:\n"
            "1. Use ONLY the provided notes context.\n"
            "2. Always infer the best possible answer from the context when any relevant information exists.\n"
            "3. Only say \"I couldn't find that in the notes yet.\" when NO context is retrieved.\n"
            "4. Always provide a bridge button based on something that IS in the notes.\n"
            "5. Keep the answer to a MAXIMUM of 4 sentences.\n"
            "6. Use the persona: {{persona_name}}.\n"
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
            "Example (if provided):\n{{example}}\n\n"
            "Previous conversation (if provided):\n{{chat_history}}\n\n"
            "Context:\n{{context}}\n\n"
            "Question:\n{{question}}"
        ),
        settings.langfuse_prompt_document_metadata: (
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
            "Filename:\n{{filename}}\n\n"
            "Document excerpt:\n{{context}}"
        ),
        settings.langfuse_prompt_quiz: (
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
            "- Create exactly {{num_questions}} questions.\n"
            "- Each question must have 4 answer options.\n"
            "- The correct_answer must exactly match one option.\n"
            "- Use only information supported by the notes context.\n"
            "- Keep wording clear and exam-oriented.\n"
            "- Prioritize reinforcement for these weak concepts when provided.\n\n"
            "Topic:\n{{topic}}\n\n"
            "Weak concepts:\n{{weak_concepts}}\n\n"
            "Context:\n{{context}}"
        ),
        settings.langfuse_prompt_reteach: (
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
            "Target Concept & Error:\n{{weak_concepts_with_error}}\n\n"
            "Context:\n{{context}}"
        ),
        settings.langfuse_prompt_study_plan: (
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
            "Reviewed concepts:\n{{reviewed_concepts}}\n\n"
            "Weak concepts:\n{{weak_concepts}}\n\n"
            "Context:\n{{context}}"
        ),
        settings.langfuse_prompt_follow_up: (
            "You are a study coach helping a learner stay engaged.\n"
            "Write one short follow-up question that encourages deeper learning.\n"
            "Start the question with: \"Would you like me to explain\".\n"
            "If the answer is incomplete because notes were missing, ask the user to share more notes.\n"
            "Return valid JSON with this shape:\n"
            '{"follow_up":"string","action":"quiz|example|explain|clarify"}\n\n'
            "Question:\n{{question}}\n\n"
            "Answer:\n{{answer}}\n\n"
            "Used fallback:\n{{used_fallback}}\n\n"
            "Context:\n{{context}}"
        ),
    }


def _prompt_exists(client: Langfuse, name: str) -> bool:
    """Check whether a Langfuse prompt already exists.

    Args:
        client (Langfuse): Langfuse client instance.
        name (str): Prompt name to check.

    Returns:
        bool: True when the prompt already exists.
    """
    try:
        client.get_prompt(name)
        return True
    except Exception:
        return False


def seed_prompts(*, force: bool, label: str) -> None:
    """Seed Langfuse prompt templates from local defaults.

    Args:
        force (bool): Whether to overwrite by creating a new version.
        label (str): Langfuse label for the prompt version.
    """
    if not langfuse_enabled():
        raise RuntimeError("Langfuse is not configured. Set LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY first.")

    configure_langfuse_environment()
    langfuse = Langfuse()
    templates = _prompt_templates()

    for name, prompt in templates.items():
        if not force and _prompt_exists(langfuse, name):  # Avoid version churn unless forced.
            print(f"Skipping {name}: already exists.")
            continue

        langfuse.create_prompt(
            name=name,
            type="text",
            prompt=prompt,
            labels=[label],
            tags=["study-pal"],
            commit_message="Seed prompt from repository defaults.",
        )
        print(f"Seeded {name}.")


def main() -> None:
    """Parse CLI args and seed Langfuse prompts."""
    parser = argparse.ArgumentParser(description="Seed Langfuse prompt templates from local defaults.")
    parser.add_argument("--force", action="store_true", help="Overwrite by creating a new prompt version.")
    parser.add_argument("--label", default="production", help="Langfuse label for the prompt version.")
    args = parser.parse_args()

    seed_prompts(force=args.force, label=args.label)


if __name__ == "__main__":
    main()
