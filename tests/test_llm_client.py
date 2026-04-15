"""test_llm_client.py: Unit tests for structured-answer lane defaults."""

from src.core.models import StructuredAnswer
from src.llm.llm_client import STRUCTURED_ANSWER_SCHEMA, _ensure_action_lanes


def test_ensure_action_lanes_adds_missing_lanes() -> None:
    """Missing info/quiz lanes should be backfilled for UI rendering."""
    structured = StructuredAnswer(
        answer="The amygdala helps process emotion.",
        citations=["anatomy_example.pdf, page 38"],
        topic_subject="Amygdala",
        info_lane=None,
        quiz_lane=None,
    )

    _ensure_action_lanes(structured, "what is the amygdala?")

    assert structured.info_lane is not None
    assert structured.quiz_lane is not None
    assert structured.info_lane.button_label.startswith("🧠")
    assert "amygdala" in structured.info_lane.query.lower()
    assert "knowledge" in structured.quiz_lane.button_label.lower()


def test_ensure_action_lanes_fills_empty_values() -> None:
    """Present but empty lane fields should be normalized to usable defaults."""
    structured = StructuredAnswer.model_validate(
        {
            "answer": "Axons carry signals.",
            "citations": ["anatomy_example.pdf, page 50"],
            "topic_subject": "Axons",
            "info_lane": {"button_label": "", "query": ""},
            "quiz_lane": {"button_label": "", "intent": "START_QUIZ_LOOP"},
        }
    )

    _ensure_action_lanes(structured, "what are axons?")

    assert structured.info_lane is not None
    assert structured.quiz_lane is not None
    assert structured.info_lane.button_label.startswith("🧠")
    assert structured.info_lane.query
    assert structured.quiz_lane.button_label


def test_structured_answer_schema_requires_lane_keys() -> None:
    """Structured-output schema should require both action lanes."""
    raw_required = STRUCTURED_ANSWER_SCHEMA.get("required", [])
    required_keys = set(raw_required) if isinstance(raw_required, list) else set()
    assert {"info_lane", "quiz_lane"}.issubset(required_keys)
