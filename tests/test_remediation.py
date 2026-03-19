from src.remediation import generate_remediation_message, is_valid_remediation_message


def test_generate_remediation_message_falls_back_without_session_context() -> None:
    message = generate_remediation_message("Photosynthesis", ["light reactions"])

    assert "Photosynthesis" in message
    assert "light reactions" in message.lower()


def test_generate_remediation_message_handles_perfect_score_case() -> None:
    message = generate_remediation_message("Photosynthesis", [])

    assert "answered everything correctly" in message


def test_remediation_validator_rejects_blank_or_too_short_output() -> None:
    assert not is_valid_remediation_message(None)
    assert not is_valid_remediation_message("Too short")
    assert is_valid_remediation_message(
        "Focus on how light reactions capture energy and support the later stages of sugar production."
    )
