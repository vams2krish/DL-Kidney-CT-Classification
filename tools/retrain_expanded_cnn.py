"""Train and evaluate a compact CNN on the project and screened CT Kidney images."""

import argparse
import hashlib
import json
import re
from contextlib import nullcontext
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Local original CTKidney.zip")
    source.add_argument("--dataset", type=Path, help="Extracted CTKidney folder")
    parser.add_argument(
        "--project", type=Path, default=Path("datasets/Project-Kidney-CT")
    )
    parser.add_argument(
        "--export", type=Path, help="Save a final CNN after cross-validation"
    )
    args = parser.parse_args()
    Path(".demo-logs").mkdir(exist_ok=True)
    root = args.project
    dataset = args.dataset or Path("datasets/CTKidney")
    labels = ["Normal", "Tumor"]
    rows = [
        (p, i)
        for i, label in enumerate(labels)
        for p in sorted(
            (root / label).glob("*.jpg"),
            key=lambda p: int(re.search(r"\((\d+)\)", p.name)[1]),
        )
    ]
    images = []
    for path, _ in rows:
        with Image.open(path) as image:
            images.append(
                np.asarray(
                    ImageOps.exif_transpose(image)
                    .convert("RGB")
                    .resize((224, 224), Image.Resampling.BILINEAR)
                )
            )
    x = np.stack(images)
    y = np.array([label for _, label in rows])
    hashes = [hashlib.sha256(image.tobytes()).digest() for image in x]
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    def prepare(rgb):
        image = Image.fromarray(rgb[12:212, 12:212]).convert("L").resize((64, 64))
        return np.asarray(image, dtype="float32")[..., None] / 255

    def dhash(rgb):
        gray = np.asarray(Image.fromarray(rgb).convert("L").resize((9, 8)))
        return int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), "big")

    def numeric(name):
        match = re.fullmatch(r"(Normal|Tumor)- \((\d+)\)\.jpg", name)
        return int(match[2]) if match else None

    local_numbers = [[(numeric(p.name)) for p, c in rows if c == i] for i in range(2)]
    local_hashes = set(hashes)
    local_visual = [dhash(rgb) for rgb in x]
    extra = []
    extra_labels = []
    rng = np.random.default_rng(42)
    source_context = ZipFile(args.archive) if args.archive else nullcontext()
    with source_context as archive:
        for label, name in enumerate(labels):
            folder = ("normal_kidney", "kidney_tumor")[label]
            if archive is not None:
                members = [
                    p for p in archive.namelist() if Path(p).parent.name == folder
                ]
            else:
                members = sorted(str(path) for path in (dataset / folder).glob("*.jpg"))
            members = [
                member for member in members if numeric(Path(member).name) is not None
            ]
            kept = 0
            for member in rng.permutation(members):
                index = numeric(Path(member).name)
                if any(abs(index - n) <= 80 for n in local_numbers[label]):
                    continue
                input_image = (
                    BytesIO(archive.read(member)) if archive is not None else member
                )
                with Image.open(input_image) as image:
                    rgb = np.asarray(
                        ImageOps.exif_transpose(image)
                        .convert("RGB")
                        .resize((224, 224), Image.Resampling.BILINEAR)
                    )
                if (np.ptp(rgb, axis=2) > 25).mean() > 0.0003:
                    continue
                digest = hashlib.sha256(rgb.tobytes()).digest()
                visual = dhash(rgb)
                if digest in local_hashes or any(
                    (visual ^ prior).bit_count() <= 2 for prior in local_visual
                ):
                    continue
                local_hashes.add(digest)
                extra.append(prepare(rgb))
                extra_labels.append(label)
                kept += 1
                if kept == 450:
                    break
            print("External images retained", name, kept, flush=True)
            assert kept >= 100, "Too few screened files for a balanced pretraining run"

    x = np.stack([prepare(rgb) for rgb in x])
    extra = np.stack(extra)
    extra_labels = np.asarray(extra_labels)

    def train_scratch(inputs, labels, train, epochs, initial_weights=None):
        layers = tf.keras.layers
        model = tf.keras.Sequential(
            [
                layers.Input((64, 64, 1)),
                layers.RandomFlip("horizontal"),
                layers.RandomTranslation(0.06, 0.06),
                layers.RandomRotation(0.04),
                layers.Conv2D(16, 5, activation="relu", padding="same"),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, activation="relu", padding="same"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, activation="relu", padding="same"),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dropout(0.3),
                layers.Dense(1),
            ]
        )
        if initial_weights is not None:
            model.set_weights(initial_weights)
        optimizer = tf.keras.optimizers.Adam(
            0.0003 if initial_weights is not None else 0.001
        )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )
        for epoch in range(epochs):
            order = np.random.default_rng(42 + epoch).permutation(train)
            for batch in np.array_split(order, max(1, len(order) // 32)):
                model.train_on_batch(inputs[batch], labels[batch])
        return model

    initial = train_scratch(
        extra, extra_labels, np.arange(len(extra)), 18
    ).get_weights()
    print("Completed external pretraining", flush=True)
    blocks = [np.array_split(np.flatnonzero(y == label), 5) for label in range(2)]
    prediction = np.empty_like(y)
    folds = []
    for fold, (normal, tumor) in enumerate(zip(*blocks)):
        test = np.concatenate([normal, tumor])
        excluded = set(test)
        for block in (normal, tumor):
            for i in (block[0] - 1, block[-1] + 1):
                if 0 <= i < len(y) and y[i] == y[block[0]]:
                    excluded.add(i)
        test_hashes = {hashes[i] for i in test}
        train = [
            i
            for i, digest in enumerate(hashes)
            if i not in excluded and digest not in test_hashes
        ]
        model = train_scratch(x, y, train, 12, initial)
        prediction[test] = (
            np.asarray(model(x[test], training=False)).ravel() > 0
        ).astype(int)
        print(
            "Fold",
            fold + 1,
            "correct",
            int((prediction[test] == y[test]).sum()),
            flush=True,
        )
        folds.append(
            {
                "fold": fold + 1,
                "train": len(train),
                "test": len(test),
                "correct": int((prediction[test] == y[test]).sum()),
            }
        )

    def stats(pred):
        return {
            "correct": int((pred == y).sum()),
            "total": len(y),
            "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        }

    report = {
        "method": "Five folds, contiguous numeric filename blocks within each class, neighboring images and training copies of exact resized test pixels excluded. Patient identifiers unavailable.",
        "labels": labels,
        "external_pretraining_images": len(extra),
        "folds": folds,
        "retrained": stats(prediction),
    }
    suffix = "" if archive is not None else "-folder"
    Path(f".demo-logs/blocked-expanded{suffix}-evaluation.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))
    if args.export:
        final_model = train_scratch(x, y, np.arange(len(y)), 12, initial)
        args.export.parent.mkdir(parents=True, exist_ok=True)
        final_model.save(args.export)
        restored = tf.keras.models.load_model(args.export, compile=False)
        np.testing.assert_allclose(
            restored(x[:4], training=False),
            final_model(x[:4], training=False),
            rtol=1e-6,
            atol=1e-6,
        )
        print("Saved and reloaded", args.export, flush=True)


if __name__ == "__main__":
    main()
