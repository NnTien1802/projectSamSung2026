"""FastAPI V1.1: original routes + opt-in M1 PyVi raw-text experiment.

Use one worker, one GPU. All checkpoints and original adapters remain frozen.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    ArticleRequest, ClickbaitRequest, ConsistencyRequest,
    RawSpellingRequest, SpellingRequest,
)
from .service import InferenceService

LOG = logging.getLogger(__name__)


def create_app(
    *, registry: Optional[Any] = None, load_models: bool = True,
    experimental_segmenter: Optional[Callable[[str], str]] = None,
) -> FastAPI:
    """Fake registry and segmenter are injectable for zero-GPU HTTP tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        if load_models:
            from system.backend.app.services.model_registry import ModelRegistry
            actual_registry = registry if registry is not None else ModelRegistry(
                device=os.environ.get("SAMSUNG_DEVICE", "cuda")
            )
            LOG.info("Initializing frozen Module 1 ...")
            actual_registry.spelling_adapter()
            LOG.info("Initializing shared M2/M3 pipeline ...")
            actual_registry.headline_adapters()
            app.state.registry = actual_registry
            app.state.service = InferenceService(
                actual_registry, app.state.gpu_lock,
                experimental_segmenter=experimental_segmenter,
            )
            app.state.ready = True
            LOG.info("All three model endpoints ready")
        elif registry is not None:
            app.state.registry = registry
            app.state.service = InferenceService(
                registry, app.state.gpu_lock,
                experimental_segmenter=experimental_segmenter,
            )
            app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            app.state.service = None
            app.state.registry = None

    api = FastAPI(
        title="projectSamSung V1 API",
        version="1.1.0",
        description="Frozen Vietnamese spelling, clickbait and headline-consistency inference",
        lifespan=lifespan, docs_url="/docs", redoc_url=None,
    )
    api.state.ready = False
    api.state.service = None
    api.state.registry = None
    api.state.gpu_lock = threading.Lock()

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @api.middleware("http")
    async def prevent_caching(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def get_service() -> InferenceService:
        if not api.state.ready or api.state.service is None:
            raise HTTPException(status_code=503, detail="Models are not ready")
        return api.state.service

    @api.get("/api/v1/health", tags=["System"])
    def health():
        return {"status": "ok" if api.state.ready else "initializing", "models_ready": bool(api.state.ready)}

    @api.get("/api/v1/models", tags=["System"])
    def models():
        return {
            "ready": bool(api.state.ready),
            "models": [
                {"module": "M1", "name": "PhoBERT MIL", "threshold": 0.57,
                 "input": "article_segmented", "ready": bool(api.state.ready),
                 "optional_raw_input": "pyvi_experimental (opt in; non-reference segmentation)"},
                {"module": "M2", "name": "mBERT Clickbait", "threshold": 0.61,
                 "input": "title", "ready": bool(api.state.ready)},
                {"module": "M3", "name": "mBERT Description + MAX-MIL", "threshold": 0.45,
                 "input": "title, description, article_text", "ready": bool(api.state.ready)},
            ],
            "shared_pipeline": ["M2", "M3"],
        }

    @api.post("/api/v1/spelling", tags=["Models"])
    def spelling(body: SpellingRequest):
        return get_service().spelling(body.article_segmented)

    @api.post("/api/v1/spelling/raw", tags=["Experimental"])
    def spelling_raw(body: RawSpellingRequest):
        return get_service().spelling_raw(body.article_text)

    @api.post("/api/v1/clickbait", tags=["Models"])
    def clickbait(body: ClickbaitRequest):
        return get_service().clickbait(body.title)

    @api.post("/api/v1/consistency", tags=["Models"])
    def consistency(body: ConsistencyRequest):
        return get_service().consistency(body.title, body.description, body.article_text)

    @api.post("/api/v1/analyze", tags=["Models"])
    def analyze(body: ArticleRequest):
        return get_service().analyze(body)

    return api


app = create_app()
