"""HTTP contract tests. Do NOT load real checkpoints or require CUDA."""
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
from system.backend.app.api.main import create_app


class FakeSpelling:
    def __init__(self, calls, delay=0):
        self.calls = calls
        self.delay = delay
        self.running = 0
        self.max_parallel = 0
        self.mutex = threading.Lock()

    def predict(self, text):
        with self.mutex:
            self.running += 1
            self.max_parallel = max(self.max_parallel, self.running)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.calls.append(("M1", text))
            return {"probability_error": 0.1, "threshold": 0.57, "prediction": 0, "label": "NO_ERROR", "chunk_count": 2}
        finally:
            with self.mutex:
                self.running -= 1


class FakeClickbait:
    def __init__(self, calls):
        self.calls = calls

    def predict(self, title):
        self.calls.append(("M2", title))
        return {"probability_clickbait": 0.2, "threshold": 0.61, "prediction": 0, "label": "NORMAL"}


class FakeConsistency:
    def __init__(self, calls):
        self.calls = calls

    def predict(self, title, description, article):
        self.calls.append(("M3", title, description, article))
        return {"probability_consistent": 0.8, "threshold": 0.45, "prediction": 1,
                "label": "CONSISTENT", "branches": {"description": {"weight": 0.5}, "max_mil": {"weight": 0.5}}}


class FakeRegistry:
    def __init__(self, delay=0):
        self.calls = []
        self.m1 = FakeSpelling(self.calls, delay=delay)
        self.m2 = FakeClickbait(self.calls)
        self.m3 = FakeConsistency(self.calls)
        self.pipeline_gets = 0

    def spelling_adapter(self):
        return self.m1

    def headline_adapters(self):
        self.pipeline_gets += 1
        return self.m2, self.m3


class APIContractTests(unittest.TestCase):
    def setUp(self):
        self.registry = FakeRegistry()
        self.client_context = TestClient(create_app(registry=self.registry, load_models=False))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_health_and_models(self):
        health = self.client.get('/api/v1/health')
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()['models_ready'])
        models = self.client.get('/api/v1/models').json()
        self.assertEqual([m['threshold'] for m in models['models']], [.57, .61, .45])
        self.assertEqual(models['shared_pipeline'], ['M2', 'M3'])
        self.assertEqual(health.headers['cache-control'], 'no-store')

    def test_unified_raw_input_skips_m1_explicitly(self):
        response = self.client.post('/api/v1/analyze', json={
            'title': 'Tin mới', 'description': 'Mô tả', 'article_text': 'Nội dung bài báo'})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body['module1_spelling']['status'], 'skipped')
        self.assertEqual(body['module2_clickbait']['result']['label'], 'NORMAL')
        self.assertEqual(body['module3_consistency']['result']['label'], 'CONSISTENT')
        self.assertEqual(self.registry.calls, [
            ('M2', 'Tin mới'), ('M3', 'Tin mới', 'Mô tả', 'Nội dung bài báo')])

    def test_unified_with_presegmented_text_runs_all_three(self):
        response = self.client.post('/api/v1/analyze', json={
            'title': 'Tin mới', 'description': 'Mô tả', 'article_text': 'Nội dung',
            'article_segmented': 'người_dân đã phát_hiện lỗi'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['module1_spelling']['result']['threshold'], .57)
        self.assertEqual(self.registry.calls[0], ('M1', 'người_dân đã phát_hiện lỗi'))
        self.assertEqual(self.registry.calls[1][0], 'M2')
        self.assertEqual(self.registry.calls[2][0], 'M3')

    def test_three_individual_routes(self):
        self.assertEqual(self.client.post('/api/v1/spelling', json={
            'article_segmented': 'người_dân'}).json()['threshold'], .57)
        self.assertEqual(self.client.post('/api/v1/clickbait', json={
            'title': 'Tiêu đề'}).json()['threshold'], .61)
        self.assertEqual(self.client.post('/api/v1/consistency', json={
            'title': 'Tiêu đề', 'article_text': 'Nội dung'}).json()['threshold'], .45)

    def test_blank_and_missing_inputs_rejected(self):
        self.assertEqual(self.client.post('/api/v1/clickbait', json={'title': '  '}).status_code, 422)
        self.assertEqual(self.client.post('/api/v1/spelling', json={}).status_code, 422)
        self.assertEqual(self.client.post('/api/v1/analyze', json={
            'title': 'Tin mới', 'article_text': '  '}).status_code, 422)
        self.assertEqual(self.registry.calls, [])

    def test_oversized_body_field_rejected(self):
        response = self.client.post('/api/v1/analyze', json={
            'title': 'Tin mới', 'article_text': 'x' * 120001})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.registry.calls, [])

    def test_failed_model_does_not_leak_exception_or_article(self):
        self.registry.m2.predict = lambda title: (_ for _ in ()).throw(RuntimeError('sensitive path'))
        response = self.client.post('/api/v1/clickbait', json={'title': 'Tin mới'})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('sensitive path', response.text)

    def test_single_gpu_lock_serializes_requests(self):
        self.client_context.__exit__(None, None, None)
        self.registry = FakeRegistry(delay=.06)
        self.client_context = TestClient(create_app(registry=self.registry, load_models=False))
        self.client = self.client_context.__enter__()

        def post(_):
            return self.client.post('/api/v1/spelling', json={'article_segmented': 'người_dân'})
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(post, range(3)))
        self.assertTrue(all(r.status_code == 200 for r in results))
        self.assertEqual(self.registry.m1.max_parallel, 1)

    def test_health_before_startup_is_not_ready(self):
        # Lifespan has not run; no fake model has been installed.
        from system.backend.app.api.main import create_app as factory
        other = factory(load_models=False)
        self.assertFalse(other.state.ready)


if __name__ == '__main__':
    unittest.main()
