"""Local research demo API. Serve with Waitress via serve.py."""
import base64
import binascii
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge, UnsupportedMediaType

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from cnnClassifer.pipeline.prediction import InvalidImage, PredictionPipeline


def create_app(classifier=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
    engine = classifier if classifier is not None else PredictionPipeline()
    app.extensions['classifier'] = engine

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; img-src 'self' data: blob:; "
            "script-src 'self'; style-src 'self'; connect-src 'self'; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get('/')
    def home():
        return render_template('index.html')

    @app.get('/health')
    def health():
        return jsonify(status='ok', model_loaded=engine.model is not None,
                       purpose='research-demo')

    @app.get('/ready')
    def ready():
        try:
            with engine.lock:
                engine.load_model()
            return jsonify(status='ready', model='VGG16')
        except Exception:
            app.logger.exception('Model readiness failed')
            return jsonify(error='Model unavailable'), 503

    @app.route('/train', methods=['GET', 'POST'])
    def train_disabled():
        return jsonify(error='Web training is disabled. Run python main.py or dvc repro locally.'), 403

    @app.post('/predict')
    def predict():
        payload = request.get_json()
        if not isinstance(payload, dict) or not isinstance(payload.get('image'), str) or not payload['image']:
            return jsonify(error='Provide a non-empty base64 image string.'), 400
        encoded = payload['image']
        if encoded.startswith('data:'):
            if ',' not in encoded or ';base64,' not in encoded:
                return jsonify(error='Invalid image data URL.'), 400
            encoded = encoded.split(',', 1)[1]
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return jsonify(error='Image must contain valid base64 data.'), 400
        if len(data) > 5 * 1024 * 1024:
            return jsonify(error='Image exceeds the 5 MB limit.'), 413
        try:
            return jsonify(engine.predict_bytes(data))
        except InvalidImage as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            app.logger.exception('Prediction failed')
            return jsonify(error='Model unavailable. Check server logs and readiness.'), 503

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(exc):
        return jsonify(error='Request exceeds the 8 MB limit.'), 413

    @app.errorhandler(BadRequest)
    def bad_request(exc):
        return jsonify(error='Request body must contain valid JSON.'), 400

    @app.errorhandler(UnsupportedMediaType)
    def wrong_type(exc):
        return jsonify(error='Use Content-Type: application/json.'), 415

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '8080')), debug=False)
