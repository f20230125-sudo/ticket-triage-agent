"""
Tests for triage.py.

Gemini is replaced by a stub client, so these need no API key and make no
network calls.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage  # noqa: E402
from triage import Category, TicketTriage, Urgency, triage_ticket  # noqa: E402

MESSAGE = "I was charged twice for the same deposit, please refund the extra charge."

SAMPLE = TicketTriage(
    category=Category.BILLING,
    urgency=Urgency.HIGH,
    summary="Customer was double charged for one deposit and wants a refund.",
    suggested_reply="Sorry about the duplicate charge. We are refunding it now.",
)


class StubClient:
    """Stands in for genai.Client: raises each queued error, then answers."""

    def __init__(self, errors=(), parsed=SAMPLE):
        self.errors = list(errors)
        self.parsed = parsed
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(parsed=self.parsed)


def server_error():
    return genai_errors.ServerError(503, {"error": {"message": "model overloaded"}})


@pytest.fixture
def sleeps(monkeypatch):
    """Record retry delays instead of actually waiting."""
    recorded = []
    monkeypatch.setattr(triage.time, "sleep", recorded.append)
    return recorded


@pytest.fixture
def stub(monkeypatch):
    """Give main() an API key and a stub client in place of the real one."""
    client = StubClient()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(triage.genai, "Client", lambda api_key: client)
    return client


def test_returns_the_parsed_triage():
    client = StubClient()

    assert triage_ticket(client, MESSAGE) == SAMPLE
    assert len(client.calls) == 1


def test_sends_the_message_and_asks_for_schema_validated_json():
    client = StubClient()

    triage_ticket(client, MESSAGE)

    call = client.calls[0]
    assert call["model"] == triage.MODEL
    assert MESSAGE in call["contents"]
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is TicketTriage


def test_schema_rejects_a_category_outside_the_enum():
    with pytest.raises(ValidationError):
        TicketTriage(category="refunds", urgency="high", summary="s", suggested_reply="r")


def test_schema_rejects_an_urgency_outside_the_enum():
    with pytest.raises(ValidationError):
        TicketTriage(category="billing", urgency="critical", summary="s", suggested_reply="r")


def test_schema_accepts_every_documented_label():
    for category in ("billing", "technical", "account", "feedback", "other"):
        for urgency in ("low", "medium", "high"):
            parsed = TicketTriage(category=category, urgency=urgency, summary="s", suggested_reply="r")
            assert parsed.category.value == category
            assert parsed.urgency.value == urgency


def test_retries_when_the_model_is_temporarily_unavailable(sleeps):
    client = StubClient(errors=[server_error(), server_error()])

    assert triage_ticket(client, MESSAGE) == SAMPLE
    assert len(client.calls) == 3
    assert sleeps == [triage.RETRY_DELAY_SECONDS] * 2


def test_retry_notice_goes_to_stderr_not_stdout(sleeps, capsys):
    client = StubClient(errors=[server_error()])

    triage_ticket(client, MESSAGE)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "retrying" in captured.err


def test_exits_clearly_when_the_reply_has_no_parsed_triage(sleeps):
    client = StubClient(parsed=None)

    with pytest.raises(SystemExit, match="no usable triage"):
        triage_ticket(client, MESSAGE)

    assert len(client.calls) == 1
    assert sleeps == []


def test_gives_up_after_the_last_retry(sleeps):
    client = StubClient(errors=[server_error() for _ in range(triage.MAX_RETRIES)])

    with pytest.raises(genai_errors.ServerError):
        triage_ticket(client, MESSAGE)

    assert len(client.calls) == triage.MAX_RETRIES
    assert len(sleeps) == triage.MAX_RETRIES - 1


def test_does_not_retry_a_bad_request(sleeps):
    bad_request = genai_errors.ClientError(400, {"error": {"message": "invalid argument"}})
    client = StubClient(errors=[bad_request])

    with pytest.raises(genai_errors.ClientError):
        triage_ticket(client, MESSAGE)

    assert len(client.calls) == 1
    assert sleeps == []


def test_main_needs_an_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["triage.py"])

    with pytest.raises(SystemExit, match="GEMINI_API_KEY is not set"):
        triage.main()


def test_main_triages_the_message_given_on_the_command_line(monkeypatch, stub, capsys):
    monkeypatch.setattr(sys, "argv", ["triage.py", "The", "app", "crashes", "on", "charts"])

    triage.main()

    assert len(stub.calls) == 1
    assert "The app crashes on charts" in stub.calls[0]["contents"]
    out = capsys.readouterr().out
    assert "Ticket:   The app crashes on charts" in out
    assert "Category: billing" in out
    assert "Urgency:  high" in out
    assert SAMPLE.suggested_reply in out


def test_main_triages_every_sample_ticket_when_given_no_argument(monkeypatch, stub):
    monkeypatch.setattr(sys, "argv", ["triage.py"])

    triage.main()

    assert len(stub.calls) == len(triage.SAMPLE_TICKETS)
    for call, ticket in zip(stub.calls, triage.SAMPLE_TICKETS):
        assert ticket in call["contents"]
