"""Compare MLPs and dummy baselines on the prepared splits."""

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SEED = 42
MAX_EPOCHS = 40
PATIENCE = 6
MIN_DELTA = 1e-5
LEARNING_RATE = 0.001


@dataclass(frozen=True)
class Architecture:
    widths: tuple[int, ...]
    activation: str = "ReLU"
    optimizer: str = "Adam"
    dropout: float = 0.0


ARCHITECTURES = {
    "MLP-1": Architecture((32,)),
    "MLP-2": Architecture((64, 32)),
    "MLP-3": Architecture((128, 64, 32)),
    "MLP-2-tanh": Architecture((64, 32), activation="Tanh"),
    "MLP-2-rmsprop": Architecture((64, 32), optimizer="RMSprop"),
    "MLP-2-dropout": Architecture((64, 32), dropout=0.15),
}


def build_model(inputs: int, outputs: int, architecture: Architecture) -> nn.Sequential:
    layers = []
    activation = {"ReLU": nn.ReLU, "Tanh": nn.Tanh}[architecture.activation]
    for width in architecture.widths:
        layers.extend((nn.Linear(inputs, width), activation()))
        if architecture.dropout:
            layers.append(nn.Dropout(architecture.dropout))
        inputs = width
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


def choose_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    valid = (thresholds > 0) & (thresholds < 1)
    if not valid.any():
        return 0.5
    quality = (tpr - fpr)[valid]
    candidates = thresholds[valid][np.isclose(quality, quality.max(), rtol=0, atol=1e-12)]
    return float(candidates[np.argmin(abs(candidates - 0.5))])


def predict(model: nn.Module, features: torch.Tensor, task: str) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        if task == "binary":
            return logits.squeeze(1).sigmoid().numpy()
        if task == "multiclass":
            return logits.softmax(dim=1).numpy()
        return logits.squeeze(1).numpy()


def evaluate(y: np.ndarray, scores: np.ndarray, task: str, threshold: float = 0.5) -> dict:
    if task == "regression":
        return {"mse": mean_squared_error(y, scores), "mae": mean_absolute_error(y, scores),
                "r2": r2_score(y, scores)}
    if task == "binary":
        predictions = (scores >= threshold).astype(int)
        labels = [0, 1]
        auc = {"1": roc_auc_score(y, scores)}
    else:
        predictions = scores.argmax(axis=1)
        labels = list(range(scores.shape[1]))
        auc = {str(label): roc_auc_score(y == label, scores[:, label]) for label in labels}
    return {
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "auc": auc,
        "macro_auc": float(np.mean(list(auc.values()))),
        "classification": classification_report(
            y, predictions, labels=labels, output_dict=True, zero_division=0,
        ),
        "confusion": confusion_matrix(y, predictions, labels=labels).tolist(),
    }


def train(
    data: dict, task: str, architecture: Architecture, batch_size: int,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(SEED)
    X_train = torch.from_numpy(data["X_train"])
    X_val = torch.from_numpy(data["X_val"])
    y_train = torch.from_numpy(data["y_train"])
    y_val = torch.from_numpy(data["y_val"])
    outputs = len(data["class_names"]) if task == "multiclass" else 1
    model = build_model(X_train.shape[1], outputs, architecture)
    if task == "binary":
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    if task != "multiclass":
        y_train = y_train.float().unsqueeze(1)
        y_val = y_val.float().unsqueeze(1)
    batch_generator = torch.Generator().manual_seed(SEED)
    optimizer_type = {"Adam": torch.optim.Adam, "RMSprop": torch.optim.RMSprop}
    optimizer = optimizer_type[architecture.optimizer](model.parameters(), lr=LEARNING_RATE)

    history = {"train": [], "val": []}
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(X_train), generator=batch_generator)
        for batch in indices.split(batch_size):
            optimizer.zero_grad()
            loss = criterion(model(X_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            # Evaluate both parts at the same weights, with dropout disabled.
            train_loss = criterion(model(X_train), y_train).item()
            val_loss = criterion(model(X_val), y_val).item()
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(
            f"  Epoch {epoch:02d}/{MAX_EPOCHS:02d} | "
            f"train loss: {train_loss:.5f} | val loss: {val_loss:.5f}",
            flush=True,
        )
        if val_loss < best_loss - MIN_DELTA:
            best_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            print(
                f"  Early stopping: no validation improvement for {PATIENCE} epochs. "
                f"Restoring epoch {best_epoch}.",
                flush=True,
            )
            break
    if best_state is None:
        raise RuntimeError("Training produced no finite validation loss")
    model.load_state_dict(best_state)
    return model, {"history": history, "best_epoch": best_epoch,
                   "parameters": sum(p.numel() for p in model.parameters())}


def plot_loss(run: dict, path: Path, *, log_scale: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    epochs = range(1, len(run["history"]["train"]) + 1)
    for split, values in run["history"].items():
        ax.plot(epochs, values, label=split)
    ax.axvline(run["best_epoch"], ls="--", color="gray", label="Selected epoch")
    ax.set(xlabel="Epoch", ylabel="Loss")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc(y: np.ndarray, scores: np.ndarray, classes: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    curves = [(1, scores)] if scores.ndim == 1 else enumerate(scores.T)
    for label, probability in curves:
        actual = y == label
        fpr, tpr, _ = roc_curve(actual, probability)
        ax.plot(fpr, tpr, label=f"{classes[label]} (AUC = {roc_auc_score(actual, probability):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random ranking")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(result: dict, classes: list[str], path: Path) -> None:
    matrices = {label: run["test"]["confusion"] for label, run in result["runs"].items()}
    matrices["Dummy"] = result["baseline"]["confusion"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 6), layout="constrained")
    upper = max(np.max(matrix) for matrix in matrices.values())
    for ax, (label, values) in zip(axes.flat, matrices.items()):
        matrix = np.asarray(values)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=upper)
        for (row, column), count in np.ndenumerate(matrix):
            color = "white" if count > 0.55 * upper else "black"
            ax.text(column, row, str(count), ha="center", va="center", color=color, fontsize=9)
        ax.set(title=label, xticks=range(len(classes)), yticks=range(len(classes)),
               xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="Actual")
    for ax in axes.flat[len(matrices):]:
        ax.set_visible(False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_experiment(name: str, task: str, batch_size: int, output: Path) -> dict:
    with np.load(ROOT / "data" / "processed" / f"{name}.npz", allow_pickle=False) as archive:
        data = dict(archive)
    X_val = torch.from_numpy(data["X_val"])
    y_val = data["y_val"]
    models = {}
    runs = {}
    for index, (label, architecture) in enumerate(ARCHITECTURES.items(), start=1):
        print(f"\n[{name} {index}/{len(ARCHITECTURES)}] Training {label}", flush=True)
        model, run = train(data, task, architecture, batch_size)
        scores = predict(model, X_val, task)
        threshold = choose_threshold(y_val, scores) if task == "binary" else 0.5
        run.update(threshold=threshold, validation=evaluate(y_val, scores, task, threshold))
        models[label] = model
        runs[label] = run
        torch.save(model.state_dict(), output / f"{name}_{label}.pt")
        print(f"{name} {label}: {len(run['history']['val'])} epochs, best {run['best_epoch']}", flush=True)

    metric = "mse" if task == "regression" else "balanced_accuracy"
    direction = min if task == "regression" else max
    winner = direction(runs, key=lambda label: runs[label]["validation"][metric])
    plot_loss(runs[winner], output / f"{name}_loss.png", log_scale=task == "regression")

    if task == "regression":
        dummy = DummyRegressor(strategy="mean")
    else:
        dummy = DummyClassifier(strategy="prior")
    dummy.fit(data["X_train"], data["y_train"])
    X_test = torch.from_numpy(data["X_test"])
    y_test = data["y_test"]
    # The winner and all binary decision thresholds are frozen before this pass.
    for label, model in models.items():
        scores = predict(model, X_test, task)
        runs[label]["test"] = evaluate(y_test, scores, task, runs[label]["threshold"])
        if label == winner and task != "regression":
            plot_roc(y_test, scores, data["class_names"].tolist(), output / f"{name}_roc.png")
    if task == "regression":
        baseline = dummy.predict(data["X_test"])
    else:
        baseline = dummy.predict_proba(data["X_test"])
        if task == "binary":
            baseline = baseline[:, 1]
    result = {"winner": winner, "batch_size": batch_size, "runs": runs,
              "baseline": evaluate(y_test, baseline, task)}
    if task != "regression":
        plot_confusion(result, data["class_names"].tolist(), output / f"{name}_confusion.png")
    return result


def main() -> None:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    results = {
        "training": {"seed": SEED, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                     "min_delta": MIN_DELTA, "learning_rate": LEARNING_RATE},
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
    }
    for name, task, batch_size in (
        ("diabetes", "binary", 1024),
        ("body_performance", "multiclass", 128),
        ("wine_quality", "regression", 128),
    ):
        results[name] = run_experiment(name, task, batch_size, output)
        print(f"{name}: winner {results[name]['winner']}", flush=True)
    (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
