from __future__ import annotations

from pathlib import Path

import pytest

from docsum.extract import UnsupportedFileType, extract_text


def test_extract_txt(tmp_path: Path):
    f = tmp_path / "notes.txt"
    f.write_text("hello world", encoding="utf-8")
    assert extract_text(f) == "hello world"


def test_extract_md(tmp_path: Path):
    f = tmp_path / "notes.md"
    f.write_text("# heading", encoding="utf-8")
    assert extract_text(f) == "# heading"


def test_unsupported_extension_raises(tmp_path: Path):
    f = tmp_path / "data.xyz"
    f.write_text("whatever", encoding="utf-8")
    with pytest.raises(UnsupportedFileType):
        extract_text(f)
