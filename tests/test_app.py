"""Boundary tests plus real bundled-model inference (no patient data)."""
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import sys
from threading import Lock

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from cnnClassifer.pipeline.prediction import PredictionPipeline


def image_bytes(color=255, mode='RGB', fmt='PNG'):
    buffer = BytesIO()
    Image.new(mode, (32, 48), color).save(buffer, format=fmt)
    return buffer.getvalue()


def payload(data):
    return {'image': base64.b64encode(data).decode()}


class FakeEngine:
    def __init__(self):
        self.model = None
        self.lock = Lock()

    def load_model(self):
        self.model = True

    def predict_bytes(self, data):
        PredictionPipeline.preprocess(data)
        return [{'image': 'Normal', 'score': .75, 'scores': {'Normal': .75, 'Tumor': .25}}]


@pytest.fixture
def client():
    return create_app(FakeEngine()).test_client()


def test_health_and_readiness(client):
    assert client.get('/health').json['model_loaded'] is False
    assert client.get('/ready').status_code == 200
    assert client.get('/health').json['model_loaded'] is True


def test_no_web_training(client):
    assert client.get('/train').status_code == 403
    assert client.post('/train').status_code == 403


@pytest.mark.parametrize('body', [None, [], {}, {'image': 42}, {'image': ''}, {'image': '!!!'}, {'image': 'data:image/png,no'}])
def test_bad_payload(client, body):
    response = client.post('/predict', json=body, content_type='application/json')
    assert response.status_code == 400
    assert 'error' in response.json


def test_wrong_content_type(client):
    assert client.post('/predict', data='hello').status_code == 415


def test_corrupt_and_unsupported_images(client):
    assert client.post('/predict', json=payload(b'not an image')).status_code == 400
    assert client.post('/predict', json=payload(image_bytes(fmt='GIF'))).status_code == 400


def test_limits(client):
    assert client.post('/predict', json=payload(b'a' * (5 * 1024 * 1024 + 1))).status_code == 413
    assert client.post('/predict', data=b' ' * (8 * 1024 * 1024 + 1), content_type='application/json').status_code == 413
    huge = BytesIO()
    Image.new('L', (4001, 4000)).save(huge, format='PNG')
    assert client.post('/predict', json=payload(huge.getvalue())).status_code == 400


def test_preprocessing_matches_training():
    batch = PredictionPipeline.preprocess(image_bytes(mode='L'))
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert np.all(batch == 1.0)
    assert np.all(PredictionPipeline.preprocess(image_bytes(color=0)) == 0.0)


@pytest.mark.parametrize('mode,fmt', [('RGB', 'PNG'), ('RGB', 'JPEG'), ('L', 'PNG'), ('RGBA', 'PNG')])
def test_valid_request_and_data_url(client, mode, fmt):
    body = payload(image_bytes(mode=mode, fmt=fmt))
    assert client.post('/predict', json=body).json[0]['score'] == .75
    body['image'] = 'data:image/png;base64,' + body['image']
    assert client.post('/predict', json=body).status_code == 200


def test_security_headers(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
    assert 'Access-Control-Allow-Origin' not in response.headers
    assert response.headers['Cache-Control'] == 'no-store'


def test_model_failure_returns_503_without_details():
    class Broken(FakeEngine):
        def load_model(self):
            raise RuntimeError('internal secret path')
        def predict_bytes(self, data):
            raise RuntimeError('internal secret path')
    client = create_app(Broken()).test_client()
    for response in [client.get('/ready'), client.post('/predict', json=payload(image_bytes()))]:
        assert response.status_code == 503
        assert b'internal secret path' not in response.data


def test_real_model_and_concurrent_requests():
    engine = PredictionPipeline()
    app = create_app(engine)
    fixture = (ROOT / 'static/synthetic-demo.png').read_bytes()
    before = (ROOT / 'inputImage.jpg').exists()
    def request_once(data):
        with app.test_client() as client:
            response = client.post('/predict', json=payload(data))
            assert response.status_code == 200
            result = response.json[0]
            assert result['image'] in {'Normal', 'Tumor'}
            assert sum(result['scores'].values()) == pytest.approx(1, abs=1e-5)
            assert result['score'] == max(result['scores'].values())
            return result
    first = request_once(fixture)
    black = request_once(image_bytes(color=0))
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(request_once, [fixture, image_bytes(color=0), fixture]))
    assert results[0]['scores'] == first['scores'] == results[2]['scores']
    assert results[1]['scores'] == black['scores']
    assert (ROOT / 'inputImage.jpg').exists() == before
