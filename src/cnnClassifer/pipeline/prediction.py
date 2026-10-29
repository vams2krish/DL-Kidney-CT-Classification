"""Validated, in-memory inference for the bundled VGG16 classifier."""
from io import BytesIO
from pathlib import Path
from threading import Lock
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[3]


class InvalidImage(ValueError):
    pass


class PredictionPipeline:
    def __init__(self, filename=None, model_path=None):
        self.filename = filename
        self.model_path = Path(model_path or ROOT / 'model' / 'model.h5')
        self.model = None
        self.lock = Lock()

    def load_model(self):
        if self.model is None:
            from tensorflow.keras.models import load_model
            self.model = load_model(self.model_path, compile=False)
        return self.model

    @staticmethod
    def preprocess(data):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as source:
                    if source.format not in {'JPEG', 'PNG'}:
                        raise InvalidImage('Choose a JPEG or PNG image.')
                    if source.width * source.height > 16_000_000:
                        raise InvalidImage('Image exceeds the 16 megapixel limit.')
                    source.load()
                    rgb = ImageOps.exif_transpose(source).convert('RGB')
                    resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
                    return np.expand_dims(np.asarray(resized, dtype=np.float32) / 255.0, 0)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise InvalidImage('The file is not a readable JPEG or PNG image.') from exc

    def predict_bytes(self, data):
        batch = self.preprocess(data)
        with self.lock:
            model = self.load_model()
            scores = np.asarray(model(batch, training=False))[0]
        if scores.shape != (2,) or not np.isfinite(scores).all():
            raise RuntimeError('Invalid classifier output')
        index = int(np.argmax(scores))
        labels = ['Normal', 'Tumor']
        return [{
            'image': labels[index],
            'score': float(scores[index]),
            'scores': {label: float(value) for label, value in zip(labels, scores)},
            'model': 'VGG16',
            'notice': 'Research demo only. Scores are uncalibrated and are not diagnostic confidence.',
        }]

    def predict(self):
        return self.predict_bytes(Path(self.filename).read_bytes())
