"""Regression tests for opt-in PyVi API. Fake models; no PyVi install/GPU needed."""
import threading
import unittest

from fastapi.testclient import TestClient
from system.backend.app.api.experimental_segmenter import segment_raw_pyvi
from system.backend.app.api.main import create_app


class FakeM1:
    def __init__(self, calls):
        self.calls = calls

    def predict(self, text):
        self.calls.append(("M1", text))
        return {"probability_error": .14, "threshold": .57, "prediction": 0,
                "label": "NO_ERROR", "chunk_count": 1}


class FakeM2:
    def __init__(self, calls):
        self.calls = calls

    def predict(self, title):
        self.calls.append(("M2", title))
        return {"probability_clickbait": .1, "threshold": .61, "label": "NORMAL"}


class FakeM3:
    def __init__(self, calls):
        self.calls = calls

    def predict(self, title, description, article):
        self.calls.append(("M3", title, description, article))
        return {"probability_consistent": .9, "threshold": .45, "label": "CONSISTENT"}


class FakeRegistry:
    def __init__(self):
        self.calls = []
        self.m1 = FakeM1(self.calls)
        self.m2 = FakeM2(self.calls)
        self.m3 = FakeM3(self.calls)

    def spelling_adapter(self):
        return self.m1

    def headline_adapters(self):
        return self.m2, self.m3


class M1PyviAPITests(unittest.TestCase):
    def setUp(self):
        self.registry = FakeRegistry()
        self.tokenizer_calls = []

        def tokenizer(text):
            self.tokenizer_calls.append(text)
            return "người_dân đi_học" if text == "người dân đi học" else text

        self.ctx = TestClient(create_app(
            registry=self.registry, load_models=False,
            experimental_segmenter=tokenizer,
        ))
        self.client = self.ctx.__enter__()

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def test_explicit_raw_route_has_warning(self):
        response = self.client.post("/api/v1/spelling/raw", json={"article_text": "người dân đi học"})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["input_mode"], "pyvi_experimental")
        self.assertIn("EXPERIMENTAL", result["warning"])
        self.assertEqual(result["result"]["threshold"], .57)
        self.assertEqual(self.registry.calls, [("M1", "người_dân đi_học")])

    def test_unified_explicit_opt_in_runs_three_models(self):
        response = self.client.post("/api/v1/analyze", json={
            "title": "Tin mới", "description": "Mô tả", "article_text": "người dân đi học",
            "m1_input_mode": "pyvi_experimental",
        })
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["module1_spelling"]["input_mode"], "pyvi_experimental")
        self.assertEqual([x[0] for x in self.registry.calls], ["M1", "M2", "M3"])
        self.assertEqual(result["module3_consistency"]["result"]["label"], "CONSISTENT")

    def test_default_raw_input_still_skips_m1(self):
        response = self.client.post("/api/v1/analyze", json={
            "title": "Tin mới", "article_text": "người dân đi học",
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["module1_spelling"]["status"], "skipped")
        self.assertEqual([x[0] for x in self.registry.calls], ["M2", "M3"])
        self.assertEqual(self.tokenizer_calls, [])

    def test_presegmented_route_stays_unchanged(self):
        response = self.client.post("/api/v1/spelling", json={
            "article_segmented": "người_dân đã đi_học",
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["threshold"], .57)
        self.assertEqual(self.registry.calls, [("M1", "người_dân đã đi_học")])
        self.assertEqual(self.tokenizer_calls, [])

    def test_reject_ambiguous_dual_input(self):
        response = self.client.post("/api/v1/analyze", json={
            "title": "Tin", "article_text": "người dân đi học",
            "article_segmented": "người_dân đi_học",
            "m1_input_mode": "pyvi_experimental",
        })
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.registry.calls, [])

    def test_raw_input_rejects_unexpected_surface_change_without_model_calls(self):
        self.ctx.__exit__(None, None, None)
        self.ctx = TestClient(create_app(
            registry=self.registry, load_models=False,
            experimental_segmenter=lambda _: "người_dân đi chơi",
        ))
        self.client = self.ctx.__enter__()
        response = self.client.post("/api/v1/spelling/raw", json={"article_text": "người dân đi học"})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("người dân đi học", response.text)
        self.assertEqual(self.registry.calls, [])

    def test_raw_input_rejects_joins_without_original_space(self):
        self.ctx.__exit__(None, None, None)
        self.ctx = TestClient(create_app(
            registry=self.registry, load_models=False,
            experimental_segmenter=lambda _: "x_y đi học",
        ))
        self.client = self.ctx.__enter__()
        response = self.client.post("/api/v1/spelling/raw", json={"article_text": "xy đi học"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.registry.calls, [])

    def test_blank_raw_rejected_and_underscore_rejected(self):
        self.assertEqual(self.client.post("/api/v1/spelling/raw", json={"article_text": "  "}).status_code, 422)
        self.assertEqual(self.client.post("/api/v1/spelling/raw", json={"article_text": "người_dân"}).status_code, 422)
        self.assertEqual(self.registry.calls, [])

    def test_invalid_mode_rejected_before_inference(self):
        response = self.client.post("/api/v1/analyze", json={
            "title": "Tin", "article_text": "người dân đi học", "m1_input_mode": "pyvi_standard",
        })
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.registry.calls, [])

    def test_segmenter_function_nfc_and_punctuation(self):
        self.assertEqual(segment_raw_pyvi("người dân, đi học", lambda _: "người_dân , đi_học"),
                         "người_dân , đi_học")
        with self.assertRaisesRegex(ValueError, "changed visible"):
            segment_raw_pyvi("quận uỷ", lambda _: "quận ủy")
        with self.assertRaisesRegex(ValueError, "does not accept underscores"):
            segment_raw_pyvi("đã_đi", lambda x: x)

    def test_segmenter_exception_has_no_user_text(self):
        def explode(_):
            raise RuntimeError("sensitive article content")
        with self.assertRaises(ValueError) as error:
            segment_raw_pyvi("người dân đi học", explode)
        self.assertNotIn("sensitive", str(error.exception))


if __name__ == "__main__":
    unittest.main()
