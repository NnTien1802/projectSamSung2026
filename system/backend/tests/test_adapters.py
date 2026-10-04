"""Fast CPU-only routing tests, not real model regression."""
import unittest
from unittest.mock import Mock
from system.backend.app.services.model_registry import ModelRegistry


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Mock()
        self.pipeline.predict_clickbait.return_value = {"label": "NORMAL", "threshold": .61}
        self.pipeline.predict_consistency.return_value = {"label": "CONSISTENT", "threshold": .45}
        self.registry = ModelRegistry(original_pipeline=self.pipeline)
        self.clickbait, self.consistency = self.registry.headline_adapters()

    def test_shared_pipeline(self):
        self.assertIs(self.clickbait.pipeline, self.consistency.pipeline)
        self.assertIs(self.registry.headline_adapters()[0].pipeline, self.pipeline)

    def test_clickbait_routes_title_only(self):
        self.assertEqual(self.clickbait.predict("  Tieu de  ")["threshold"], .61)
        self.pipeline.predict_clickbait.assert_called_once_with(title="Tieu de")

    def test_consistency_routes_three_fields(self):
        self.consistency.predict("Tieu de", "Mo ta", "Noi dung")
        self.pipeline.predict_consistency.assert_called_once_with(title="Tieu de", lead="Mo ta", content="Noi dung")

    def test_empty_description_warns(self):
        result = self.consistency.predict("Tieu de", None, "Noi dung")
        self.assertTrue(result["warnings"])
        self.pipeline.predict_consistency.assert_called_once_with(title="Tieu de", lead="", content="Noi dung")

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            self.clickbait.predict(" ")
        with self.assertRaises(ValueError):
            self.consistency.predict("Tieu de", "Mo ta", " ")


if __name__ == "__main__":
    unittest.main()
