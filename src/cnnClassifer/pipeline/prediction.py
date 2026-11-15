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


ATTENTION_COLORS = {
    'Normal': (20, 184, 166),
    'Tumor': (251, 113, 133),
    'Stone': (245, 158, 11),
    'Cyst': (167, 139, 250),
}


class PredictionPipeline:
    name = 'VGG16'
    labels = ('Normal', 'Tumor')

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
        if scores.shape != (len(self.labels),) or not np.isfinite(scores).all():
            raise RuntimeError('Invalid classifier output')
        index = int(np.argmax(scores))
        return [{
            'image': self.labels[index],
            'score': float(scores[index]),
            'scores': {label: float(value) for label, value in zip(self.labels, scores)},
            'model': self.name,
            'notice': 'Research demo only. Scores are uncalibrated and are not diagnostic confidence.',
        }]

    def predict(self):
        return self.predict_bytes(Path(self.filename).read_bytes())

    def explain_bytes(self, data, class_index):
        """Describe the input and visualize positive evidence for the selected class."""
        with Image.open(BytesIO(data)) as source:
            image_format = source.format
            monochrome = source.mode in {'1', 'L', 'I', 'I;16', 'F'}
            rgb = ImageOps.exif_transpose(source).convert('RGB')
            tone = 'Grayscale' if monochrome else 'Color-encoded'
            description = f'{tone} {image_format}, {rgb.width} × {rgb.height} pixels.'
        explanation = {'description': description}
        if not self.supports_attention(class_index):
            return explanation

        import tensorflow as tf
        batch = tf.convert_to_tensor(self.preprocess(data))
        with self.lock:
            model = self.load_model()
            head = model.layers[-1]
            probe = tf.keras.Model(model.input, [
                self.attention_layer(model).output, head.input,
            ])
            with tf.GradientTape() as tape:
                if batch.dtype.is_floating:
                    tape.watch(batch)
                activations, features = probe(batch, training=False)
                logits = features @ head.kernel
                if head.bias is not None:
                    logits += head.bias
                other_classes = [index for index in range(len(self.labels)) if index != class_index]
                margin = logits[:, class_index] - tf.reduce_mean(
                    tf.gather(logits, other_classes, axis=1), axis=1,
                )
            gradients = tape.gradient(margin, activations)
        if gradients is None:
            return explanation
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        heat = tf.maximum(tf.reduce_sum(activations * weights, axis=-1)[0], 0).numpy()
        maximum = heat.max()
        if not np.isfinite(maximum) or maximum <= 0:
            return explanation

        source_size = rgb.size
        rgb.thumbnail((512, 512))
        mask = Image.fromarray(np.uint8(heat / maximum * 150))
        label = self.labels[class_index]
        color = Image.new('RGBA', rgb.size, (*ATTENTION_COLORS[label], 0))
        color.putalpha(self.align_attention(mask, source_size, rgb.size))
        with BytesIO() as buffer:
            Image.alpha_composite(rgb.convert('RGBA'), color).save(buffer, format='PNG')
            explanation['attention_image'] = base64.b64encode(buffer.getvalue()).decode('ascii')
            explanation['attention_class'] = label
        return explanation

    def supports_attention(self, class_index):
        return class_index == 1

    @staticmethod
    def attention_layer(model):
        return model.get_layer('block5_conv3')

    @staticmethod
    def align_attention(mask, source_size, preview_size):
        return mask.resize(preview_size, Image.Resampling.BILINEAR)


class FourClassPipeline(PredictionPipeline):
    name = 'Four-class CNN'
    labels = ('Normal', 'Tumor', 'Stone', 'Cyst')

    @staticmethod
    def preprocess(data):
        try:
            with Image.open(BytesIO(data)) as source:
                if source.format not in {'JPEG', 'PNG'}:
                    raise InvalidImage('Choose a JPEG or PNG image.')
                if source.width * source.height > 16_000_000:
                    raise InvalidImage('Image exceeds the 16 megapixel limit.')
                gray = ImageOps.exif_transpose(source).convert('L')
                prepared = ImageOps.pad(gray, (64, 64), Image.Resampling.BILINEAR, color=0)
                return np.asarray(prepared)[None, ..., None]
        except (OSError, Image.DecompressionBombError) as exc:
            raise InvalidImage('The file is not a readable JPEG or PNG image.') from exc

    def supports_attention(self, class_index):
        return 0 <= class_index < len(self.labels)

    @staticmethod
    def attention_layer(model):
        from tensorflow.keras.layers import Conv2D
        return next(layer for layer in reversed(model.layers) if isinstance(layer, Conv2D))

    @staticmethod
    def align_attention(mask, source_size, preview_size):
        # The input was letterboxed. Remove that padding before overlaying.
        source_width, source_height = source_size
        scale = 64 / max(source_size)
        width, height = round(source_width * scale), round(source_height * scale)
        left, top = round((64 - width) / 2), round((64 - height) / 2)
        square = mask.resize((64, 64), Image.Resampling.BILINEAR)
        return square.crop((left, top, left + width, top + height)).resize(
            preview_size, Image.Resampling.BILINEAR,
        )
