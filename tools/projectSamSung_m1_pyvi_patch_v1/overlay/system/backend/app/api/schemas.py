"""API input contracts. Frozen models and model thresholds are unchanged."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ArticleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=1000, description="Vietnamese news headline")
    description: str = Field(default="", max_length=5000, description="Article lead/summary")
    article_text: str = Field(..., min_length=1, max_length=120000, description="Full raw article")
    article_segmented: Optional[str] = Field(
        default=None, min_length=1, max_length=120000,
        description="Optional presegmented M1 input; always takes the standard M1 path."
    )
    m1_input_mode: Literal["segmented_only", "pyvi_experimental"] = Field(
        default="segmented_only",
        description="Defaults to the original API. Select pyvi_experimental explicitly to run M1 on article_text."
    )


class SpellingRequest(BaseModel):
    article_segmented: str = Field(
        ..., min_length=1, max_length=120000,
        description="Presegmented Vietnamese text (e.g., người_dân); not raw text."
    )


class RawSpellingRequest(BaseModel):
    article_text: str = Field(
        ..., min_length=1, max_length=120000,
        description="Ordinary Vietnamese article; experimental PyVi segmentation is explicitly opted into by calling this endpoint."
    )


class ClickbaitRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=1000)


class ConsistencyRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=1000)
    description: str = Field(default="", max_length=5000)
    article_text: str = Field(..., min_length=1, max_length=120000)
