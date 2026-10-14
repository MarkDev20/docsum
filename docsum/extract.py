"""Plain-text extraction from the input file. Keeps I/O separate from the
LLM logic so the rest of the pipeline never has to know whether the source
was a .txt or a .pdf.
"""

from __future__ import annotations

from pathlib import Path


class UnsupportedFileType(Exception):
    pass


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".txt" or suffix == ".md":
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Reading PDFs requires pypdf. Install it with: pip install pypdf"
            ) from exc

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    raise UnsupportedFileType(
        f"Don't know how to read '{suffix}' files. Supported: .txt, .md, .pdf"
    )
