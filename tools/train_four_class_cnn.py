"""Train a four-class kidney CT research CNN with filename-block evaluation."""

import argparse
import hashlib
import json
import re
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = {
    "Normal": "normal_kidney",
    "Tumor": "kidney_tumor",
    "Stone": "kidney_stone",
    "Cyst": "cyst_kidney",
}


def numeric_id(path):
    match = re.search(r"\((\d+)\)", path.stem)
    if not match:
        raise ValueError(f"Missing numeric filename ID: {path.name}")
    return int(match[1])


def load_split(root, archive=None):
    images, labels, split = [], [], []
    for label, folder in enumerate(CLASSES.values()):
        if archive is None:
            paths = (root / folder).glob("*.jpg")
        else:
            paths = (
                Path(member)
                for member in archive.namelist()
                if Path(member).parent.name == folder and member.endswith(".jpg")
            )
        paths = sorted(paths, key=numeric_id)
        if len(paths) < 100:
            raise ValueError(f"Expected a full class folder at {root / folder}")
        # Contiguous blocks cover early, middle, and late parts of each class.
        # Drop both boundary images of each block to reduce adjacent-slice overlap.
        for block, indices in enumerate(np.array_split(np.arange(len(paths)), 15)):
            partition = (
                "test"
                if block % 5 == 4
                else "validation"
                if block % 5 == 2
                else "train"
            )
            for index in indices[1:-1]:
                path = paths[index]
                source = (
                    path if archive is None else BytesIO(archive.read(path.as_posix()))
                )
                with Image.open(source) as image:
                    image = ImageOps.exif_transpose(image).convert("L")
                    image = ImageOps.pad(
                        image, (64, 64), Image.Resampling.BILINEAR, color=0
                    )
                    images.append(np.asarray(image))
                labels.append(label)
                split.append(partition)
        print("Loaded", folder, len(paths), flush=True)

    images = np.stack(images)[..., None]
    labels = np.asarray(labels)
    split = np.asarray(split)
    digests = np.asarray(
        [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    )
    # Drop transformed-pixel duplicates, prioritizing test, then validation.
    seen = set()
    partitions = {}
    for partition in ("test", "validation", "train"):
        indices = []
        for index in np.flatnonzero(split == partition):
            if digests[index] not in seen:
                indices.append(index)
                seen.add(digests[index])
        indices = np.asarray(indices)
        if set(labels[indices]) != set(range(len(CLASSES))):
            raise ValueError(f"Every class must occur in {partition}")
        partitions[partition] = (images[indices], labels[indices])
    return partitions


def build_model():
    layers = tf.keras.layers
    return tf.keras.Sequential(
        [
            layers.Input((64, 64, 1), dtype="uint8"),
            layers.Rescaling(1 / 255),
            layers.RandomTranslation(0.06, 0.06),
            layers.RandomRotation(0.035),
            layers.Conv2D(16, 5, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(48, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(len(CLASSES), activation="softmax"),
        ],
        name="kidney_ct_four_class",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/CTKidney"))
    parser.add_argument(
        "--archive", type=Path, help="Read the same CTKidney data directly from its ZIP"
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/four-class-ct-cnn.keras")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/four-class-evaluation.json")
    )
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    with ZipFile(args.archive) if args.archive else nullcontext() as archive:
        partitions = load_split(args.dataset, archive)
    train_x, train_y = partitions["train"]
    validation_x, validation_y = partitions["validation"]
    test_x, test_y = partitions["test"]
    model = build_model()
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    class_counts = np.bincount(train_y)
    weights = np.sqrt(len(train_y) / (len(CLASSES) * class_counts))
    history = model.fit(
        train_x,
        train_y,
        sample_weight=weights[train_y],
        validation_data=(validation_x, validation_y),
        batch_size=64,
        epochs=args.epochs,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )
        ],
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    restored = tf.keras.models.load_model(args.output, compile=False)
    probabilities = restored.predict(test_x, batch_size=64, verbose=0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    predicted = probabilities.argmax(axis=1)
    report = {
        "classes": list(CLASSES),
        "method": "15 contiguous filename blocks per class, boundary slices and exact resized duplicates removed. Patient identifiers unavailable. Checkpoint selected on validation loss.",
        "split_counts": {
            name: dict(zip(CLASSES, np.bincount(y, minlength=len(CLASSES)).tolist()))
            for name, (_, y) in partitions.items()
        },
        "best_epoch": best_epoch,
        "test_accuracy": float(np.mean(predicted == test_y)),
        "test_majority_baseline": float(np.bincount(test_y).max() / len(test_y)),
        "confusion_matrix": confusion_matrix(
            test_y, predicted, labels=range(len(CLASSES))
        ).tolist(),
        "classification_report": classification_report(
            test_y,
            predicted,
            target_names=list(CLASSES),
            output_dict=True,
            zero_division=0,
        ),
        "model_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
