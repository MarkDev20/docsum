"""
Structured output schema for the document summary.

Why this exists: an LLM, left alone, returns prose. For a summary to be
usable by other code (stored in a DB, shown in a UI, fed to another step),
it has to come back as data with a known shape. This module defines that
shape and validates whatever the model returns against it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

__all__ = ["DocumentSummary", "ValidationError"]


class DocumentSummary(BaseModel):
    """The structured summary we ask the model to produce."""

    title: str = Field(..., description="A short descriptive title for the document.")
    summary: str = Field(..., description="A 2-4 sentence plain-language summary.")
    key_points: list[str] = Field(
        default_factory=list,
        description="The most important facts or claims, as short bullet-style strings.",
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Concrete follow-up actions implied by the document, if any.",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Named people, companies, dates or amounts worth tracking.",
    )

    # This is what we hand the model as the output contract. Keeping it next
    # to the schema it describes means they can never drift apart silently.
    @classmethod
    def json_schema_for_prompt(cls) -> str:
        import json

        return json.dumps(cls.model_json_schema(), indent=2)
