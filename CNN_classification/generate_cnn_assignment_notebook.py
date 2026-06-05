import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "cnn_classification_assignment.ipynb"
REPORT_PATH = ROOT / "reports" / "final_report_template.md"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


cells = [
    md(
        r"""
# CNN Classification Lab: Accuracy Is Not Enough

This notebook implements the full assignment workflow:

1. A CNN trained from scratch on CIFAR-10.
2. Imbalanced classification on DermaMNIST with metrics beyond accuracy.
3. Transfer learning and Grad-CAM analysis on Oxford-IIIT Pet.

Run this notebook in Google Colab with a GPU runtime. The code is intentionally organized so that experiments can run quickly for development and then be scaled by increasing the epoch constants in the configuration cell.
"""
    ),
    md(
        r"""
## Report: Introduction

Image classification accuracy can hide important failure modes. This lab compares balanced classification, imbalanced medical-style classification, and transfer learning to show why validation/test separation, class-wise metrics, confusion matrices, and visual explanations are necessary.

The test set is held out until the final evaluation for each scenario. Model choices use only training and validation evidence.
"""
    ),
    code(
        r"""
# If running in a fresh Colab runtime, install the small medical image dataset package.
import sys, subprocess, importlib.util

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

if importlib.util.find_spec("medmnist") is None:
    pip_install("medmnist")
"""
    ),
    code(
        r"""
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow_datasets as tfds
from medmnist import INFO
from medmnist.dataset import DermaMNIST

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
"""
    ),
    code(
        r"""
@dataclass
class ExperimentConfig:
    batch_size: int = 64
    cifar_epochs: int = 15
    derma_epochs: int = 12
    pets_head_epochs: int = 6
    pets_finetune_epochs: int = 6
    fast_run: bool = False

CFG = ExperimentConfig()

if CFG.fast_run:
    CFG.cifar_epochs = 2
    CFG.derma_epochs = 2
    CFG.pets_head_epochs = 1
    CFG.pets_finetune_epochs = 1

AUTOTUNE = tf.data.AUTOTUNE
"""
    ),
    code(
        r"""
def plot_class_distribution(labels, class_names, title):
    labels = np.asarray(labels).reshape(-1)
    counts = pd.Series(labels).value_counts().sort_index()
    df = pd.DataFrame({"class": [class_names[i] for i in counts.index], "count": counts.values})
    plt.figure(figsize=(max(8, len(class_names) * 0.55), 4))
    sns.barplot(data=df, x="class", y="count", color="#4C78A8")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return df


def plot_history(history, title):
    hist = pd.DataFrame(history.history)
    metrics = [c for c in hist.columns if not c.startswith("val_")]
    fig, axes = plt.subplots(1, min(2, len(metrics)), figsize=(12, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, metric in zip(axes, metrics[:2]):
        ax.plot(hist[metric], label=f"train {metric}")
        val_metric = f"val_{metric}"
        if val_metric in hist:
            ax.plot(hist[val_metric], label=f"val {metric}")
        ax.set_title(f"{title}: {metric}")
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.show()


def predict_labels(model, x, batch_size=128):
    probs = model.predict(x, batch_size=batch_size, verbose=0)
    return probs, np.argmax(probs, axis=1)


def evaluate_predictions(y_true, y_pred, class_names, title, y_prob=None):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    metrics = {
        "model": title,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    print(pd.Series(metrics).to_string())
    print("\nClassification report")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(7, len(class_names) * 0.55), max(5, len(class_names) * 0.45)))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot=False)
    plt.title(f"{title}: confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    return metrics, cm


def show_examples(images, y_true, y_pred, y_prob, class_names, title, selector, max_items=9):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    confidence = np.max(y_prob, axis=1)
    idx = selector(y_true, y_pred, confidence)
    idx = np.asarray(idx)[:max_items]
    if len(idx) == 0:
        print(f"No examples found for {title}")
        return
    cols = 3
    rows = int(np.ceil(len(idx) / cols))
    plt.figure(figsize=(cols * 3.2, rows * 3.2))
    for pos, i in enumerate(idx, 1):
        plt.subplot(rows, cols, pos)
        img = images[i]
        if img.max() <= 1:
            plt.imshow(img)
        else:
            plt.imshow(img.astype("uint8"))
        plt.axis("off")
        plt.title(
            f"T: {class_names[y_true[i]]}\nP: {class_names[y_pred[i]]}\nconf={confidence[i]:.2f}",
            fontsize=9,
        )
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def summarize_confusions(cm, class_names, top_n=5):
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    pairs = []
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            if cm2[i, j] > 0:
                pairs.append((cm2[i, j], class_names[i], class_names[j]))
    return pd.DataFrame(sorted(pairs, reverse=True)[:top_n], columns=["count", "true", "predicted"])
"""
    ),
    md("## Scenario 1: CNN From Scratch on CIFAR-10"),
    md(
        r"""
### Dataset Exploration

CIFAR-10 is a balanced 10-class dataset of 32 by 32 RGB images. The validation split is carved out of the original training data. The test set is not used for training, early stopping, model selection, architecture changes, or hyperparameter choices.
"""
    ),
    code(
        r"""
cifar_class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

(x_train_full, y_train_full), (x_test_cifar, y_test_cifar) = keras.datasets.cifar10.load_data()
y_train_full = y_train_full.reshape(-1)
y_test_cifar = y_test_cifar.reshape(-1)

x_train_cifar, x_val_cifar, y_train_cifar, y_val_cifar = train_test_split(
    x_train_full,
    y_train_full,
    test_size=5000,
    random_state=SEED,
    stratify=y_train_full,
)

x_train_cifar = x_train_cifar.astype("float32") / 255.0
x_val_cifar = x_val_cifar.astype("float32") / 255.0
x_test_cifar = x_test_cifar.astype("float32") / 255.0

print(x_train_cifar.shape, x_val_cifar.shape, x_test_cifar.shape)
plot_class_distribution(y_train_cifar, cifar_class_names, "CIFAR-10 training class distribution")

plt.figure(figsize=(10, 4))
for i in range(12):
    plt.subplot(2, 6, i + 1)
    plt.imshow(x_train_cifar[i])
    plt.title(cifar_class_names[y_train_cifar[i]], fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
### Baseline CNN

The baseline uses two convolutional blocks, ReLU activations, pooling, a dense classifier, and a softmax output.
"""
    ),
    code(
        r"""
def build_cifar_baseline():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="cifar_baseline_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

cifar_baseline = build_cifar_baseline()
cifar_baseline.summary()
print("Parameters:", cifar_baseline.count_params())
"""
    ),
    code(
        r"""
cifar_baseline_history = cifar_baseline.fit(
    x_train_cifar,
    y_train_cifar,
    validation_data=(x_val_cifar, y_val_cifar),
    epochs=CFG.cifar_epochs,
    batch_size=CFG.batch_size,
    verbose=1,
)
plot_history(cifar_baseline_history, "CIFAR baseline")

cifar_baseline_prob, cifar_baseline_pred = predict_labels(cifar_baseline, x_test_cifar)
cifar_baseline_metrics, cifar_baseline_cm = evaluate_predictions(
    y_test_cifar, cifar_baseline_pred, cifar_class_names, "CIFAR baseline", cifar_baseline_prob
)
"""
    ),
    md(
        r"""
### Improved CNN

The improved model uses data augmentation, batch normalization, dropout, L2 weight decay, early stopping, and learning-rate scheduling.
"""
    ),
    code(
        r"""
def build_cifar_improved():
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.08),
        ],
        name="cifar_augmentation",
    )
    wd = 1e-4
    inputs = keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(wd))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(wd))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="cifar_improved_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

cifar_improved = build_cifar_improved()
cifar_improved.summary()
print("Parameters:", cifar_improved.count_params())
"""
    ),
    code(
        r"""
cifar_callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
]

cifar_improved_history = cifar_improved.fit(
    x_train_cifar,
    y_train_cifar,
    validation_data=(x_val_cifar, y_val_cifar),
    epochs=CFG.cifar_epochs,
    batch_size=CFG.batch_size,
    callbacks=cifar_callbacks,
    verbose=1,
)
plot_history(cifar_improved_history, "CIFAR improved")

cifar_improved_prob, cifar_improved_pred = predict_labels(cifar_improved, x_test_cifar)
cifar_improved_metrics, cifar_improved_cm = evaluate_predictions(
    y_test_cifar, cifar_improved_pred, cifar_class_names, "CIFAR improved", cifar_improved_prob
)

cifar_comparison = pd.DataFrame([cifar_baseline_metrics, cifar_improved_metrics])
cifar_comparison["overfitting_gap"] = [
    max(cifar_baseline_history.history["accuracy"]) - max(cifar_baseline_history.history["val_accuracy"]),
    max(cifar_improved_history.history["accuracy"]) - max(cifar_improved_history.history["val_accuracy"]),
]
cifar_comparison[["model", "accuracy", "macro_f1", "overfitting_gap"]]
"""
    ),
    code(
        r"""
show_examples(
    x_test_cifar,
    y_test_cifar,
    cifar_improved_pred,
    cifar_improved_prob,
    cifar_class_names,
    "CIFAR correct predictions",
    lambda yt, yp, conf: np.where(yt == yp)[0][np.argsort(-conf[yt == yp])],
)
show_examples(
    x_test_cifar,
    y_test_cifar,
    cifar_improved_pred,
    cifar_improved_prob,
    cifar_class_names,
    "CIFAR high-confidence mistakes",
    lambda yt, yp, conf: np.where(yt != yp)[0][np.argsort(-conf[yt != yp])],
)
summarize_confusions(cifar_improved_cm, cifar_class_names)
"""
    ),
    md(
        r"""
### Scenario 1 Discussion

Answer after running the notebook:

1. The most frequently confused classes are listed in the confusion-pair table above.
2. Compare the training-validation accuracy gap to decide whether the improved model reduced overfitting.
3. Accuracy is useful but incomplete because it hides class-specific errors and high-confidence mistakes.
4. Training from scratch gives architectural control and no pretraining-domain assumptions, but it usually needs more data, more compute, and stronger regularization.
"""
    ),
    md("## Scenario 2: Imbalanced Classification on DermaMNIST"),
    code(
        r"""
derma_info = INFO["dermamnist"]
derma_class_names = [derma_info["label"][str(i)] for i in range(len(derma_info["label"]))]
print(derma_class_names)

derma_train = DermaMNIST(split="train", download=True)
derma_val = DermaMNIST(split="val", download=True)
derma_test = DermaMNIST(split="test", download=True)

x_train_derma = np.asarray(derma_train.imgs).astype("float32") / 255.0
y_train_derma = np.asarray(derma_train.labels).reshape(-1)
x_val_derma = np.asarray(derma_val.imgs).astype("float32") / 255.0
y_val_derma = np.asarray(derma_val.labels).reshape(-1)
x_test_derma = np.asarray(derma_test.imgs).astype("float32") / 255.0
y_test_derma = np.asarray(derma_test.labels).reshape(-1)

print(x_train_derma.shape, x_val_derma.shape, x_test_derma.shape)
derma_dist = plot_class_distribution(y_train_derma, derma_class_names, "DermaMNIST training class distribution")

majority_count = derma_dist["count"].max()
minority_count = derma_dist["count"].min()
majority_class = derma_dist.loc[derma_dist["count"].idxmax(), "class"]
minority_class = derma_dist.loc[derma_dist["count"].idxmin(), "class"]
majority_baseline_accuracy = majority_count / derma_dist["count"].sum()
imbalance_ratio = majority_count / minority_count

print(f"Majority class: {majority_class}")
print(f"Minority class: {minority_class}")
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
print(f"Majority-class baseline accuracy: {majority_baseline_accuracy:.3f}")
"""
    ),
    code(
        r"""
def build_derma_cnn(name="derma_cnn", focal=False):
    inputs = keras.Input(shape=x_train_derma.shape[1:])
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(len(derma_class_names), activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=name)
    loss = sparse_categorical_focal_loss if focal else "sparse_categorical_crossentropy"
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    return model


def sparse_categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
    y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
    p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
    ce = -tf.math.log(p_t)
    focal_factor = tf.pow(1.0 - p_t, gamma)
    return alpha * focal_factor * ce


def train_and_evaluate_derma(model, method_name, class_weight=None, x_train=None, y_train=None):
    history = model.fit(
        x_train if x_train is not None else x_train_derma,
        y_train if y_train is not None else y_train_derma,
        validation_data=(x_val_derma, y_val_derma),
        epochs=CFG.derma_epochs,
        batch_size=CFG.batch_size,
        class_weight=class_weight,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=1,
    )
    plot_history(history, method_name)
    prob, pred = predict_labels(model, x_test_derma)
    metrics, cm = evaluate_predictions(y_test_derma, pred, derma_class_names, method_name, prob)
    per_class_recall = recall_score(y_test_derma, pred, labels=list(range(len(derma_class_names))), average=None, zero_division=0)
    metrics["min_class_recall"] = float(np.min(per_class_recall))
    metrics["mean_per_class_recall"] = float(np.mean(per_class_recall))
    return model, history, prob, pred, metrics, cm, pd.DataFrame({"class": derma_class_names, "recall": per_class_recall})
"""
    ),
    code(
        r"""
derma_baseline = build_derma_cnn("derma_baseline")
derma_baseline.summary()
(
    derma_baseline,
    derma_baseline_history,
    derma_baseline_prob,
    derma_baseline_pred,
    derma_baseline_metrics,
    derma_baseline_cm,
    derma_baseline_recall,
) = train_and_evaluate_derma(derma_baseline, "Derma baseline")
derma_baseline_recall
"""
    ),
    code(
        r"""
classes = np.unique(y_train_derma)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_derma)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
class_weight
"""
    ),
    code(
        r"""
derma_weighted = build_derma_cnn("derma_class_weighted")
(
    derma_weighted,
    derma_weighted_history,
    derma_weighted_prob,
    derma_weighted_pred,
    derma_weighted_metrics,
    derma_weighted_cm,
    derma_weighted_recall,
) = train_and_evaluate_derma(derma_weighted, "Derma class-weighted loss", class_weight=class_weight)
derma_weighted_recall
"""
    ),
    code(
        r"""
def make_oversampled_data(x, y):
    counts = pd.Series(y).value_counts()
    max_count = counts.max()
    selected = []
    rng = np.random.default_rng(SEED)
    for label in sorted(counts.index):
        label_idx = np.where(y == label)[0]
        sampled = rng.choice(label_idx, size=max_count, replace=True)
        selected.append(sampled)
    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return x[selected], y[selected]

x_train_derma_over, y_train_derma_over = make_oversampled_data(x_train_derma, y_train_derma)
plot_class_distribution(y_train_derma_over, derma_class_names, "Oversampled DermaMNIST class distribution")

derma_oversampled = build_derma_cnn("derma_oversampled")
(
    derma_oversampled,
    derma_oversampled_history,
    derma_oversampled_prob,
    derma_oversampled_pred,
    derma_oversampled_metrics,
    derma_oversampled_cm,
    derma_oversampled_recall,
) = train_and_evaluate_derma(
    derma_oversampled,
    "Derma random oversampling",
    x_train=x_train_derma_over,
    y_train=y_train_derma_over,
)
derma_oversampled_recall
"""
    ),
    code(
        r"""
derma_results = pd.DataFrame(
    [derma_baseline_metrics, derma_weighted_metrics, derma_oversampled_metrics]
)
derma_results[
    ["model", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "macro_f1", "min_class_recall"]
].sort_values("macro_f1", ascending=False)
"""
    ),
    md(
        r"""
### Scenario 2 Discussion

Answer after running the notebook:

1. Accuracy can be misleading because a classifier can score well by favoring the majority class while failing rare classes.
2. Balanced accuracy, macro F1, and per-class recall better capture real performance on imbalanced data.
3. Balancing improves fairness if minority-class recall rises without unacceptable collapse in other classes.
4. In a medical context, false negatives for dangerous lesions are especially dangerous, but false positives also create cost and anxiety.
5. This model is not suitable for clinical deployment because it lacks external validation, calibration, expert review, robustness testing, and clinical workflow evaluation.
"""
    ),
    md("## Scenario 3: Transfer Learning and Grad-CAM on Oxford-IIIT Pet"),
    code(
        r"""
PETS_IMG_SIZE = 160
PETS_BATCH_SIZE = 32

def preprocess_pet(example):
    image = tf.image.resize(example["image"], (PETS_IMG_SIZE, PETS_IMG_SIZE))
    image = tf.cast(image, tf.float32)
    label = tf.cast(example["label"], tf.int32)
    return image, label

pet_builder = tfds.builder("oxford_iiit_pet")
pet_builder.download_and_prepare()
pet_info = pet_builder.info
pet_class_names = pet_info.features["label"].names
num_pet_classes = len(pet_class_names)

pet_train_raw = pet_builder.as_dataset(split="train[:85%]", shuffle_files=True)
pet_val_raw = pet_builder.as_dataset(split="train[85%:]", shuffle_files=False)
pet_test_raw = pet_builder.as_dataset(split="test", shuffle_files=False)

pet_train = (
    pet_train_raw
    .map(preprocess_pet, num_parallel_calls=AUTOTUNE)
    .shuffle(2048, seed=SEED)
    .batch(PETS_BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
pet_val = pet_val_raw.map(preprocess_pet, num_parallel_calls=AUTOTUNE).batch(PETS_BATCH_SIZE).prefetch(AUTOTUNE)
pet_test = pet_test_raw.map(preprocess_pet, num_parallel_calls=AUTOTUNE).batch(PETS_BATCH_SIZE).prefetch(AUTOTUNE)

print("Pet classes:", num_pet_classes)
plt.figure(figsize=(10, 6))
for i, (image, label) in enumerate(pet_train_raw.take(9).map(preprocess_pet)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(tf.cast(image, tf.uint8))
    plt.title(pet_class_names[int(label)], fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
pet_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
    ],
    name="pet_augmentation",
)

def build_pet_transfer_model():
    base = keras.applications.MobileNetV2(
        input_shape=(PETS_IMG_SIZE, PETS_IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    inputs = keras.Input(shape=(PETS_IMG_SIZE, PETS_IMG_SIZE, 3))
    x = pet_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_pet_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="pet_mobilenetv2_transfer")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base

pet_model, pet_base = build_pet_transfer_model()
pet_model.summary()
print("Frozen parameters:", np.sum([np.prod(v.shape) for v in pet_model.non_trainable_weights]))
print("Trainable parameters:", np.sum([np.prod(v.shape) for v in pet_model.trainable_weights]))
"""
    ),
    code(
        r"""
pet_head_history = pet_model.fit(
    pet_train,
    validation_data=pet_val,
    epochs=CFG.pets_head_epochs,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
    verbose=1,
)
plot_history(pet_head_history, "Oxford Pets transfer head")

pet_head_prob = pet_model.predict(pet_test, verbose=0)
y_test_pet = np.concatenate([y.numpy() for _, y in pet_test], axis=0)
pet_head_pred = np.argmax(pet_head_prob, axis=1)
pet_head_metrics, pet_head_cm = evaluate_predictions(
    y_test_pet, pet_head_pred, pet_class_names, "Pets transfer head", pet_head_prob
)
"""
    ),
    code(
        r"""
# Fine-tuning: unfreeze the last part of the pretrained network and reduce the learning rate.
pet_base.trainable = True
for layer in pet_base.layers[:-30]:
    layer.trainable = False

pet_model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Frozen parameters after unfreezing:", np.sum([np.prod(v.shape) for v in pet_model.non_trainable_weights]))
print("Trainable parameters after unfreezing:", np.sum([np.prod(v.shape) for v in pet_model.trainable_weights]))

pet_finetune_history = pet_model.fit(
    pet_train,
    validation_data=pet_val,
    epochs=CFG.pets_finetune_epochs,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
    verbose=1,
)
plot_history(pet_finetune_history, "Oxford Pets fine-tuning")

pet_finetune_prob = pet_model.predict(pet_test, verbose=0)
pet_finetune_pred = np.argmax(pet_finetune_prob, axis=1)
pet_finetune_metrics, pet_finetune_cm = evaluate_predictions(
    y_test_pet, pet_finetune_pred, pet_class_names, "Pets fine-tuned", pet_finetune_prob
)

pets_comparison = pd.DataFrame([pet_head_metrics, pet_finetune_metrics])
pets_comparison[["model", "accuracy", "macro_f1", "balanced_accuracy"]]
"""
    ),
    md("### Grad-CAM Visual Debugging"),
    code(
        r"""
def get_mobilenet_base(model):
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    raise ValueError("MobileNetV2 base model not found.")


pet_base_for_gradcam = get_mobilenet_base(pet_model)
last_conv_layer_name = "Conv_1"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    base_model = get_mobilenet_base(model)
    grad_base = keras.Model(
        base_model.inputs,
        [base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )
    classifier_layers = model.layers[model.layers.index(base_model) + 1 :]
    with tf.GradientTape() as tape:
        x = model.get_layer("pet_augmentation")(img_array, training=False)
        x = keras.applications.mobilenet_v2.preprocess_input(x)
        conv_outputs, x = grad_base(x)
        for layer in classifier_layers:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
        predictions = x
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.45):
    heatmap = tf.image.resize(heatmap[..., np.newaxis], image.shape[:2]).numpy().squeeze()
    cmap = plt.cm.jet(heatmap)[..., :3]
    image_float = image.astype("float32") / 255.0
    overlay = np.clip((1 - alpha) * image_float + alpha * cmap, 0, 1)
    return overlay


pet_test_images = np.concatenate([x.numpy() for x, _ in pet_test], axis=0)
pet_conf = np.max(pet_finetune_prob, axis=1)

gradcam_groups = {
    "correct high-confidence": np.where(pet_finetune_pred == y_test_pet)[0][np.argsort(-pet_conf[pet_finetune_pred == y_test_pet])[:3]],
    "incorrect high-confidence": np.where(pet_finetune_pred != y_test_pet)[0][np.argsort(-pet_conf[pet_finetune_pred != y_test_pet])[:3]],
    "low-confidence": np.argsort(pet_conf)[:3],
}

for group_name, indices in gradcam_groups.items():
    if len(indices) == 0:
        print(f"No examples for {group_name}")
        continue
    plt.figure(figsize=(10, 3 * len(indices)))
    for row, idx in enumerate(indices):
        image = pet_test_images[idx].astype("uint8")
        img_array = np.expand_dims(image.astype("float32"), axis=0)
        pred_label = int(pet_finetune_pred[idx])
        heatmap = make_gradcam_heatmap(img_array, pet_model, last_conv_layer_name, pred_label)
        overlay = overlay_heatmap(image, heatmap)

        plt.subplot(len(indices), 2, row * 2 + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"{group_name}\nTrue: {pet_class_names[int(y_test_pet[idx])]}\n"
            f"Pred: {pet_class_names[pred_label]} ({pet_conf[idx]:.2f})",
            fontsize=9,
        )
        plt.subplot(len(indices), 2, row * 2 + 2)
        plt.imshow(overlay)
        plt.axis("off")
        plt.title("Grad-CAM heatmap", fontsize=9)
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        r"""
### Scenario 3 Discussion

Answer after running the notebook:

1. Transfer learning usually outperforms training from scratch when the dataset is moderate-sized and the pretrained domain contains useful edges, textures, shapes, and object-part features.
2. Low-level visual features transfer most broadly; higher-level class-specific features transfer less reliably.
3. Fine-tuning improves performance only if it adapts the representation without overfitting.
4. Grad-CAM should increase trust only when the highlighted regions match meaningful animal features, not background artifacts.
5. A model can be correct for the wrong reason if it relies on background, pose, lighting, watermark, or other shortcut signals.
6. Grad-CAM is coarse, class-discriminative rather than causal, and does not prove that the model used only the highlighted region.
"""
    ),
    md("## Final Comparison and Reflection"),
    code(
        r"""
final_comparison = pd.DataFrame(
    [
        {
            "scenario": "Scenario 1",
            "dataset": "CIFAR-10",
            "best_model": "Improved CNN from scratch",
            "main_challenge": "Generalization on small natural images",
            "best_metric": "Macro F1 plus confusion matrix",
            "accuracy": cifar_improved_metrics["accuracy"],
            "macro_f1": cifar_improved_metrics["macro_f1"],
        },
        {
            "scenario": "Scenario 2",
            "dataset": "DermaMNIST",
            "best_model": derma_results.sort_values("macro_f1", ascending=False).iloc[0]["model"],
            "main_challenge": "Severe class imbalance",
            "best_metric": "Balanced accuracy, macro F1, minority recall",
            "accuracy": derma_results.sort_values("macro_f1", ascending=False).iloc[0]["accuracy"],
            "macro_f1": derma_results.sort_values("macro_f1", ascending=False).iloc[0]["macro_f1"],
        },
        {
            "scenario": "Scenario 3",
            "dataset": "Oxford-IIIT Pet",
            "best_model": pets_comparison.sort_values("macro_f1", ascending=False).iloc[0]["model"],
            "main_challenge": "Fine-grained visual similarity",
            "best_metric": "Macro F1 plus Grad-CAM inspection",
            "accuracy": pets_comparison.sort_values("macro_f1", ascending=False).iloc[0]["accuracy"],
            "macro_f1": pets_comparison.sort_values("macro_f1", ascending=False).iloc[0]["macro_f1"],
        },
    ]
)
final_comparison
"""
    ),
    md(
        r"""
### Final Reflection Questions

Complete these answers using the observed tables and figures:

1. Which model would you trust most?
2. Which model had the highest accuracy?
3. Was the highest-accuracy model actually the best?
4. Which scenario demonstrated the limitations of accuracy most clearly?
5. What did confusion matrices reveal that scalar metrics hid?
6. Did balancing improve fairness across classes?
7. What exactly is transferred during transfer learning?
8. Did Grad-CAM reveal shortcut learning?
9. Can a model be right for the wrong reason?
10. What additional tests would be necessary before deployment in a real-world application?
"""
    ),
    md(
        r"""
## Report: Conclusion

This notebook demonstrates that accuracy alone is insufficient. Balanced datasets still require confusion-matrix analysis, imbalanced datasets require macro and class-wise metrics, and transfer learning requires visual debugging to check whether the model attends to meaningful regions. Final conclusions should be based on the test-set metrics, confusion patterns, and Grad-CAM examples produced above.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(
    """# Final Report Template: CNN Classification Lab

## 1. Introduction
Explain why the assignment studies accuracy, imbalance, transfer learning, and visual debugging together.

## 2. Methodology
Describe the train/validation/test separation, the CNN architectures, imbalance methods, transfer learning setup, fine-tuning, and Grad-CAM procedure.

## 3. Results
Paste the final metric tables from the notebook.

## 4. Error Analysis
Summarize confusion matrices, minority-class recall, high-confidence mistakes, and Grad-CAM examples.

## 5. Discussion
Answer the required scenario questions and final reflection questions.

## 6. Conclusion
State which model you would trust most and why, including the remaining deployment risks.
""",
    encoding="utf-8",
)

print(f"Wrote {NOTEBOOK_PATH}")
print(f"Wrote {REPORT_PATH}")
