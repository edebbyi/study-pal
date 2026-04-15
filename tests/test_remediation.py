"""test_remediation.py: Tests for test_remediation.py."""

from src.modes.remediation import generate_remediation_message, is_valid_remediation_message


def test_generate_remediation_message_falls_back_without_session_context() -> None:
    """Test generate remediation message falls back without session context.
    """

    message, payload = generate_remediation_message("Photosynthesis", ["light reactions"])

    assert "Photosynthesis" in message
    assert "light reactions" in message.lower()
    assert payload is None or isinstance(payload, dict)


def test_generate_remediation_message_handles_perfect_score_case() -> None:
    """Test generate remediation message handles perfect score case.
    """

    message, payload = generate_remediation_message("Photosynthesis", [])

    assert "answered everything correctly" in message
    assert payload is None


def test_remediation_validator_rejects_blank_or_too_short_output() -> None:
    """Test remediation validator rejects blank or too short output.
    """

    assert not is_valid_remediation_message(None)
    assert not is_valid_remediation_message("Too short")
    assert is_valid_remediation_message(
        "Focus on how light reactions capture energy and support the later stages of sugar production."
    )
