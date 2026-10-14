"""
Tests for the summarization pipeline. None of these hit the real API —
we fake the client's response, the same way you'd fake an external HTTP
call in a Laravel feature test with Http::fake().

This is the part most people skip when "playing with LLM APIs", and
it's exactly what distinguishes a prototype from something you'd trust
in production: can you prove the retry logic and validation actually
work, without spending API credits or depending on the model behaving
on any given test run?
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from docsum.schema import DocumentSummary
from docsum.summarize import SummarizationError, _extract_json, summarize_document


def make_fake_response(text: str):
    """Builds an object shaped like the OpenAI-compatible SDK's
    ChatCompletion response, just enough for our code to read
    `.choices[0].message.content`."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


VALID_JSON = json.dumps(
    {
        "title": "Q3 Signing Desk Rollout",
        "summary": "Describes the first phase of the digital signature module.",
        "key_points": ["Three functional areas", "Polymorphic data model"],
        "action_items": ["Confirm scope of qualified signatures for phase 2"],
        "entities": ["SigningDesk", "Obras/Adjudicação"],
    }
)


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json(VALID_JSON)["title"] == "Q3 Signing Desk Rollout"

    def test_json_wrapped_in_markdown_fence(self):
        fenced = f"```json\n{VALID_JSON}\n```"
        assert _extract_json(fenced)["title"] == "Q3 Signing Desk Rollout"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("not json at all")


class TestSummarizeDocument:
    def test_succeeds_on_first_valid_response(self):
        client = MagicMock()
        client.chat.completions.create.return_value = make_fake_response(VALID_JSON)

        result = summarize_document(client, "some document text")

        assert isinstance(result.summary, DocumentSummary)
        assert result.summary.title == "Q3 Signing Desk Rollout"
        assert result.attempts == 1
        client.chat.completions.create.assert_called_once()

    def test_retries_after_invalid_json_then_succeeds(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_fake_response("I'm sorry, here is a summary: ..."),  # not JSON
            make_fake_response(VALID_JSON),  # corrected on retry
        ]

        result = summarize_document(client, "some document text")

        assert result.attempts == 2
        assert client.chat.completions.create.call_count == 2
        # the retry prompt should tell the model what went wrong
        second_call_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert "Your previous response was not valid JSON" in str(second_call_kwargs)

    def test_retries_after_schema_mismatch(self):
        missing_title = json.dumps({"summary": "no title field here"})
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_fake_response(missing_title),
            make_fake_response(VALID_JSON),
        ]

        result = summarize_document(client, "some document text")

        assert result.attempts == 2
        assert result.summary.title == "Q3 Signing Desk Rollout"

    def test_gives_up_after_exhausting_retries(self):
        client = MagicMock()
        client.chat.completions.create.return_value = make_fake_response("still not json")

        with pytest.raises(SummarizationError):
            summarize_document(client, "some document text")

        # MAX_VALIDATION_RETRIES=2 means 3 total attempts
        assert client.chat.completions.create.call_count == 3


class TestDocumentSummarySchema:
    def test_defaults_to_empty_lists(self):
        s = DocumentSummary(title="T", summary="S")
        assert s.key_points == []
        assert s.action_items == []
        assert s.entities == []

    def test_requires_title_and_summary(self):
        with pytest.raises(ValidationError):
            DocumentSummary(summary="only summary, no title")
