"""M3 logical adapter. Same model, weights and chunking as the original pipeline."""

class HeadlineConsistencyAdapter:
    def __init__(self, shared_pipeline):
        self.pipeline = shared_pipeline

    def predict(self, title: str, description: str | None, article_text: str) -> dict:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title is required")
        if not isinstance(article_text, str) or not article_text.strip():
            raise ValueError("article_text is required")
        if description is not None and not isinstance(description, str):
            raise TypeError("description must be a string or None")
        result = self.pipeline.predict_consistency(
            title=title.strip(), lead=(description or "").strip(), content=article_text.strip()
        )
        if not description or not description.strip():
            result = {**result, "warnings": ["description_missing: empty description branch input"]}
        return result
