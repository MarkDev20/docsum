"""
Core logic: send document text to an LLM API and get back a validated
DocumentSummary.

This uses Google's Gemini API through its OpenAI-compatible endpoint, so
it runs on Gemini's free tier (no payment method required). The rest of
the pipeline (schema, validation, retries) is provider-agnostic — if you
later want to point this at the Anthropic API, OpenAI, or anything else,
only this file changes.

Three things this module exists to teach/demonstrate, because they're
exactly what separates "I called an LLM API" from "I built something
reliable on top of one":

1. Structured output isn't free. You ask for JSON in the prompt, but the
   model can still return prose, markdown-fenced JSON, or JSON that's
   syntactically valid but doesn't match your schema. Always parse AND
   validate, never just parse.

2. Retries need to feed the error back. A blind retry just asks the same
   question again and often gets the same wrong answer. A useful retry
   tells the model what was wrong with its last attempt.

3. Errors from the API itself (rate limits, timeouts, overload) are a
   different failure mode from "the model answered badly", and need
   different handling (backoff, not re-prompting).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from openai import APIError, APIStatusError, NotFoundError, OpenAI
from pydantic import ValidationError

from .schema import DocumentSummary

# Gemini's free tier, reached through its OpenAI-compatible endpoint.
# https://ai.google.dev/gemini-api/docs/openai
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Providers deprecate model names over time (this one already changed once
# during development of this project). Hardcoding it means editing source
# code every time that happens. Reading it from an env var with a fallback
# means updating it is a config change, not a code change — the normal way
# this is handled in production.
DEFAULT_MODEL = "gemini-3.8-flash"
MODEL = os.environ.get("DOCSUM_MODEL", DEFAULT_MODEL)

MAX_VALIDATION_RETRIES = 2
MAX_API_RETRIES = 3

SYSTEM_PROMPT = """\
You are a precise document analyst. You read a document and extract a \
structured summary. You respond with ONLY a single JSON object matching \
the schema you are given — no prose before or after, no markdown code \
fences. If a field doesn't apply (e.g. no action items), return an empty \
list for it rather than omitting it or inventing content."""


def make_client(api_key: str) -> OpenAI:
    """Builds the API client. Swapping providers later means changing
    base_url/MODEL here — summarize_document() below never needs to know."""
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)


class SummarizationError(Exception):
    """Raised when we exhaust retries without getting a valid summary."""


@dataclass
class SummarizeResult:
    summary: DocumentSummary
    attempts: int
    raw_response: str


def _build_user_prompt(document_text: str, schema_json: str, retry_note: str = "") -> str:
    prompt = (
        "Summarize the following document according to this JSON schema:\n\n"
        f"{schema_json}\n\n"
    )
    if retry_note:
        prompt += f"NOTE: {retry_note}\n\n"
    prompt += f"Document:\n---\n{document_text}\n---"
    return prompt


def _extract_json(raw: str) -> dict:
    """The model is instructed to return bare JSON, but models sometimes
    wrap it in ```json fences anyway. Strip that defensively rather than
    trusting the instruction blindly."""
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


def _call_with_backoff(client: OpenAI, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except NotFoundError as exc:
            # The model name itself is wrong/retired — retrying won't help,
            # and the raw traceback doesn't make that obvious. Translate it
            # into a message that points at the actual fix.
            raise SummarizationError(
                f"Model '{kwargs.get('model')}' was not found or is no longer "
                "available. Providers retire model names over time. Check the "
                "current model list in the provider's docs and set DOCSUM_MODEL "
                f"in your .env to the current name.\nOriginal error: {exc}"
            ) from exc
        except APIStatusError as exc:
            # 429 (rate limited) and 5xx (overloaded) are worth retrying;
            # 4xx like bad request are not — retrying won't fix a bad prompt.
            last_exc = exc
            if exc.status_code == 429 or exc.status_code >= 500:
                wait = 2**attempt
                time.sleep(wait)
                continue
            raise
        except APIError as exc:  # network-level errors
            last_exc = exc
            time.sleep(2**attempt)
    raise SummarizationError(f"API call failed after {MAX_API_RETRIES} attempts: {last_exc}")


def summarize_document(client: OpenAI, document_text: str) -> SummarizeResult:
    schema_json = DocumentSummary.json_schema_for_prompt()
    retry_note = ""
    last_raw = ""

    for attempt in range(1, MAX_VALIDATION_RETRIES + 2):  # +1 for the first try
        user_prompt = _build_user_prompt(document_text, schema_json, retry_note)

        response = _call_with_backoff(
            client,
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        last_raw = raw

        try:
            data = _extract_json(raw)
            summary = DocumentSummary.model_validate(data)
            return SummarizeResult(summary=summary, attempts=attempt, raw_response=raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            retry_note = (
                "Your previous response was not valid JSON matching the schema. "
                f"Error: {exc}. Return ONLY the corrected JSON object."
            )
            continue

    raise SummarizationError(
        f"Model did not return a valid structured summary after "
        f"{MAX_VALIDATION_RETRIES + 1} attempts. Last response:\n{last_raw}"
    )
