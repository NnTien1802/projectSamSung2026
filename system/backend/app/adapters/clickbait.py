"""M2 logical adapter. Delegates to the immutable original unified pipeline."""

class ClickbaitAdapter:
    def __init__(self, shared_pipeline):
        self.pipeline = shared_pipeline

    def predict(self, title: str) -> dict:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title is required")
        return self.pipeline.predict_clickbait(title=title.strip())
