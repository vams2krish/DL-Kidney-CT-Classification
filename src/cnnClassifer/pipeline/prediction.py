"""Validated, in-memory inference for the bundled VGG16 classifier."""
import base64
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

    def explain_bytes(self, data, class_index):
        """Describe the input and optionally visualize positive Tumor-class evidence."""
        with Image.open(BytesIO(data)) as source:
            image_format = source.format
            monochrome = source.mode in {'1', 'L', 'I', 'I;16', 'F'}
            rgb = ImageOps.exif_transpose(source).convert('RGB')
            tone = 'Grayscale' if monochrome else 'Color-encoded'
            description = f'{tone} {image_format}, {rgb.width} × {rgb.height} pixels.'
        explanation = {'description': description}
        if class_index != 1:
            return explanation

        import tensorflow as tf
        batch = tf.convert_to_tensor(self.preprocess(data))
        with self.lock:
            model = self.load_model()
            head = model.layers[-1]
            probe = tf.keras.Model(model.input, [
                model.get_layer('block5_conv3').output, head.input,
            ])
            with tf.GradientTape() as tape:
                tape.watch(batch)
                activations, features = probe(batch, training=False)
                margin = features @ (head.kernel[:, 1:2] - head.kernel[:, 0:1])
            gradients = tape.gradient(margin, activations)
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        heat = tf.maximum(tf.reduce_sum(activations * weights, axis=-1)[0], 0).numpy()
        maximum = heat.max()
        if not np.isfinite(maximum) or maximum <= 0:
            return explanation

        rgb.thumbnail((512, 512))
        mask = Image.fromarray(np.uint8(heat / maximum * 150))
        color = Image.new('RGBA', rgb.size, (255, 70, 0, 0))
        color.putalpha(mask.resize(rgb.size, Image.Resampling.BILINEAR))
        with BytesIO() as buffer:
            Image.alpha_composite(rgb.convert('RGBA'), color).save(buffer, format='PNG')
            explanation['attention_image'] = base64.b64encode(buffer.getvalue()).decode('ascii')
        return explanation
