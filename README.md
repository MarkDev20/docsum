# docsum

A small command-line tool that reads a document (`.txt`, `.md` or `.pdf`)
and returns a **structured, validated** summary using an LLM API: a
title, a plain-language summary, key points, action items and named
entities — as JSON, not prose.

This is a learning project built to practice the parts of working with
LLM APIs that go beyond "send a prompt, get text back":

- **Structured output.** The model is asked for JSON matching a schema
  (`docsum/schema.py`, defined with Pydantic), and the response is parsed
  and *validated* against it — not just assumed to be correct.
- **Retries that use the error.** If the model's response isn't valid
  JSON, or doesn't match the schema, the retry prompt tells it exactly
  what was wrong, instead of blindly asking the same question again.
- **Separating failure modes.** An API-level failure (rate limit,
  timeout, server overload) is handled with exponential backoff. A
  model-level failure (bad output) is handled with a corrective retry.
  These are different problems and are handled differently in
  `docsum/summarize.py`.
- **Tests that don't touch the network.** The test suite fakes the
  API client's response (see `tests/test_summarize.py`), so it runs in
  CI without an API key, without cost, and without depending on the
  model behaving the same way twice.

## Provider

Runs on **Google's Gemini API**, free tier, via its OpenAI-compatible
endpoint — no payment method required to get a key. The provider is
isolated to a single function (`make_client` in `docsum/summarize.py`),
so swapping to the Anthropic API, OpenAI, or any other OpenAI-compatible
provider later is a one-function change; the schema, validation, retry
logic, CLI and tests don't need to know or care which provider is behind
the client.

## Why these choices

**Why Pydantic for the schema instead of just trusting the prompt?**
Because "the model was told to return JSON" and "the model returned valid
JSON matching my schema" are different guarantees. Models sometimes wrap
JSON in markdown fences, omit a field, or return the wrong type for a
field. Validating catches all of that before it reaches the rest of the
program.

**Why a corrective retry instead of just failing immediately?**
In practice, most structured-output failures are the model making a small
mistake (a stray code fence, a missing field) that it corrects
immediately when told what was wrong. A single corrective retry resolves
the large majority of these without the complexity of a more elaborate
retry strategy.

**Why separate `extract.py`, `schema.py`, `summarize.py` and `cli.py`?**
So each piece can be tested and reasoned about on its own: file reading
doesn't know about the API, the API logic doesn't know about argparse,
and the schema is the single source of truth both the prompt and the
validator read from.

## Setup

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
cp .env.example .env    # Windows: Copy-Item .env.example .env
```

Get a **free** API key (no card required) at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey), then
paste it into `.env` as `GEMINI_API_KEY`.

## Usage

```bash
python -m docsum sample_document.txt
python -m docsum sample_document.txt --out summary.json
```

Example output:

```json
{
  "title": "Digital Signature Module — Project Kickoff",
  "summary": "Notes from the kickoff of a transversal digital signature module for the ERP...",
  "key_points": [
    "Phase 1 MVP covers only the Obras / Adjudicação document type",
    "Three functional areas: configuration, global dashboard, contextual block",
    "Signers can be fixed, role-based, or resolved dynamically from the document"
  ],
  "action_items": [
    "Prototype the e-signature provider's API in isolation before full implementation",
    "Confirm whether the blocking rule is needed in phase 1",
    "Schedule a follow-up review in two weeks"
  ],
  "entities": ["Obras", "Adjudicação", "HR representative"]
}
```

## Running the tests

```bash
pytest -v
```

All tests run offline — no API key required.

## What's deliberately out of scope here

This is a learning project, kept small on purpose:

- No vector database / retrieval — that's the next project (RAG over a
  document set instead of a single file).
- No tool use / function calling — that's the project after this one
  (an agent that queries structured data instead of summarizing text).
- No persistent storage — output goes to a file or stdout.

## Stack

Python · [Gemini API](https://ai.google.dev/gemini-api/docs/openai) (OpenAI-compatible endpoint) · Pydantic · pypdf · pytest
