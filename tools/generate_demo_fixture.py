"""Generate an abstract CT-like software fixture, with no patient data."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

size = 512
y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
rng = np.random.default_rng(42)
body = (x / .73)**2 + (y / .84)**2 < 1
pixels = np.where(body, 70 + rng.normal(0, 5, (size, size)), 12)
for cx in [-.28, .28]:
    mask = ((x-cx)/.16)**2 + ((y-.06)/.23)**2 < 1
    pixels[mask] = 120 + rng.normal(0, 8, mask.sum())
spine = (x/.10)**2 + ((y-.40)/.12)**2 < 1
pixels[spine] = 205
image = Image.fromarray(np.clip(pixels, 0, 255).astype('uint8')).convert('RGB')
draw = ImageDraw.Draw(image)
draw.rectangle((0, 0, 512, 34), fill=(14, 35, 31))
draw.text((14, 12), 'SYNTHETIC SOFTWARE FIXTURE - NOT PATIENT DATA', fill=(204, 236, 216))
image.save(Path(__file__).resolve().parents[1] / 'static' / 'synthetic-demo.png')
