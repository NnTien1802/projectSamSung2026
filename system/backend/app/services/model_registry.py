"""Shared/lazy model owner. Do not create a pipeline per request or per adapter."""
from ..adapters._source import ROOT, import_original
from ..adapters.clickbait import ClickbaitAdapter
from ..adapters.headline_consistency import HeadlineConsistencyAdapter
from ..adapters.spelling import SpellingAdapter


class ModelRegistry:
    def __init__(self, device=None, original_pipeline=None, spelling_adapter=None):
        self.device = device
        self._original_pipeline = original_pipeline
        self._spelling_adapter = spelling_adapter

    def headline_adapters(self):
        if self._original_pipeline is None:
            src = ROOT / "module2" / "inference" / "headline_integrity_pipeline.py"
            original = import_original(src, "_project_samsung_original_headline_pipeline_v1")
            self._original_pipeline = original.HeadlineIntegrityPipeline(device=self.device)
        return ClickbaitAdapter(self._original_pipeline), HeadlineConsistencyAdapter(self._original_pipeline)

    def spelling_adapter(self):
        if self._spelling_adapter is None:
            self._spelling_adapter = SpellingAdapter(device=self.device)
        return self._spelling_adapter
