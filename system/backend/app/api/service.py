"""Application orchestration: serialized GPU inference, optional PyVi on raw M1.

Default V1 behavior remains identical: M1 is skipped without presegmented
input unless the caller *explicitly* opts into PyVi experimental mode.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from fastapi import HTTPException

from .experimental_segmenter import segment_raw_pyvi
from .schemas import ArticleRequest

LOG = logging.getLogger(__name__)
T = TypeVar("T")

EXPERIMENTAL_WARNING = (
    "EXPERIMENTAL: M1 received PyVi-segmented raw text, not training-compatible "
    "reference segmentation. In a 60-sample validation check, 8 predictions "
    "changed relative to the original segmentation. Not for official model evaluation."
)


def required_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=422, detail=f"{field} must not be blank")
    return value.strip()


class InferenceService:
    def __init__(
        self, registry: Any, gpu_lock: threading.Lock, wait_seconds: int = 30,
        experimental_segmenter: Optional[Callable[[str], str]] = None,
    ):
        self.registry = registry
        self.gpu_lock = gpu_lock
        self.wait_seconds = wait_seconds
        self.experimental_segmenter = experimental_segmenter

    def _locked(self, fn: Callable[[], T]) -> T:
        if not self.gpu_lock.acquire(timeout=self.wait_seconds):
            raise HTTPException(
                status_code=503,
                detail="Inference worker is busy; try again later",
                headers={"Retry-After": "3"},
            )
        try:
            return fn()
        except ValueError as exc:
            # Segmenter errors are fixed strings, never a portion of user text.
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except ImportError:
            LOG.error("Optional experimental segmenter unavailable")
            raise HTTPException(status_code=503, detail="Experimental segmentation is unavailable") from None
        except HTTPException:
            raise
        except Exception as exc:
            LOG.error("Inference failed: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Model inference failed") from None
        finally:
            self.gpu_lock.release()

    def _segment_raw(self, article: str) -> str:
        return segment_raw_pyvi(article, tokenizer=self.experimental_segmenter)

    def _predict_raw_m1(self, article: str) -> dict:
        segmented = self._segment_raw(article)
        return {
            "status": "completed",
            "input_mode": "pyvi_experimental",
            "warning": EXPERIMENTAL_WARNING,
            "result": self.registry.spelling_adapter().predict(segmented),
        }

    def spelling(self, text: str) -> dict:
        segmented = required_text(text, "article_segmented")
        return self._locked(lambda: self.registry.spelling_adapter().predict(segmented))

    def spelling_raw(self, text: str) -> dict:
        article = required_text(text, "article_text")
        return self._locked(lambda: self._predict_raw_m1(article))

    def clickbait(self, title: str) -> dict:
        clean_title = required_text(title, "title")
        return self._locked(lambda: self.registry.headline_adapters()[0].predict(clean_title))

    def consistency(self, title: str, description: str, article_text: str) -> dict:
        clean_title = required_text(title, "title")
        clean_article = required_text(article_text, "article_text")
        return self._locked(lambda: self.registry.headline_adapters()[1].predict(
            clean_title, description, clean_article
        ))

    def analyze(self, data: ArticleRequest) -> dict:
        title = required_text(data.title, "title")
        article = required_text(data.article_text, "article_text")
        segmented = data.article_segmented
        if segmented is not None:
            segmented = required_text(segmented, "article_segmented")
        if data.m1_input_mode == "pyvi_experimental" and segmented is not None:
            raise HTTPException(
                status_code=422,
                detail="Choose article_segmented OR m1_input_mode=pyvi_experimental, not both",
            )

        def run():
            if segmented is not None:
                m1 = {
                    "status": "completed",
                    "input_mode": "presegmented",
                    "result": self.registry.spelling_adapter().predict(segmented),
                }
            elif data.m1_input_mode == "pyvi_experimental":
                m1 = self._predict_raw_m1(article)
            else:
                m1 = {
                    "status": "skipped",
                    "reason": "article_segmented required by default; to opt in to experimental raw M1 set m1_input_mode=pyvi_experimental",
                }
            m2, m3 = self.registry.headline_adapters()
            return {
                "api_version": "v1",
                "module1_spelling": m1,
                "module2_clickbait": {"status": "completed", "result": m2.predict(title)},
                "module3_consistency": {"status": "completed", "result": m3.predict(title, data.description, article)},
                "notes": [
                    "Module outputs are independent; no aggregate news-quality score is calculated.",
                    "M1 reports document-level error likelihood; it does not localize or correct words.",
                ],
            }

        # One lock covers segmentation and all three inference stages.
        return self._locked(run)
