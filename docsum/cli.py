"""Command-line entry point.

Usage:
    python -m docsum path/to/document.pdf
    python -m docsum path/to/notes.txt --out summary.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .extract import extract_text
from .summarize import SummarizationError, make_client, summarize_document

# Loads GEMINI_API_KEY (and anything else) from a .env file in the current
# directory into the environment, so os.environ.get() below can see it.
load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docsum",
        description="Summarize a document into structured JSON using an LLM API.",
    )
    parser.add_argument("file", type=Path, help="Path to a .txt, .md or .pdf file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the resulting JSON (default: print to stdout)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "GEMINI_API_KEY is not set. Create a .env from .env.example, "
            "or export it in your shell. Get a free key at "
            "https://aistudio.google.com/apikey (no payment method required).",
            file=sys.stderr,
        )
        return 1

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    try:
        text = extract_text(args.file)
    except Exception as exc:
        print(f"Could not read '{args.file}': {exc}", file=sys.stderr)
        return 1

    if not text.strip():
        print(f"'{args.file}' produced no extractable text.", file=sys.stderr)
        return 1

    client = make_client(api_key)

    try:
        result = summarize_document(client, text)
    except SummarizationError as exc:
        print(f"Summarization failed: {exc}", file=sys.stderr)
        return 1

    output = result.summary.model_dump_json(indent=2)

    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"Wrote summary to {args.out} ({result.attempts} attempt(s))")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
