"""Train and evaluate the independent future-corrective-takeover risk MLP."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from evo_rlt.core.corrective_risk import (
    CorrectiveTakeoverRiskMLP,
    load_corrective_risk_checkpoint,
    masked_risk_bce_with_logits,
    save_corrective_risk_checkpoint,
)


class _RiskDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError("risk dataset split is empty")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.rows[index]
        return {
            "state": torch.as_tensor(row["state_vec"], dtype=torch.float32),
            "action": torch.as_tensor(row["action_chunk"], dtype=torch.float32),
            "label": torch.as_tensor(row["label"], dtype=torch.float32),
            "mask": torch.as_tensor(row["label_mask"], dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_counts(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    positive = negative = masked = 0
    for row in rows:
        if float(torch.as_tensor(row["label_mask"]).item()) <= 0.5:
            masked += 1
        elif float(torch.as_tensor(row["label"]).item()) > 0.5:
            positive += 1
        else:
            negative += 1
    return positive, negative, masked


def _require_two_classes(rows: list[dict[str, Any]], split: str) -> tuple[int, int, int]:
    positive, negative, masked = _label_counts(rows)
    if positive == 0 or negative == 0:
        raise ValueError(
            f"{split} split requires nonzero eligible positive and negative classes; "
            f"positive={positive}, negative={negative}, masked={masked}"
        )
    return positive, negative, masked


def binary_classification_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    logits = logits.detach().cpu().reshape(-1).to(torch.float64)
    labels = labels.detach().cpu().reshape(-1).to(torch.float64)
    if logits.numel() == 0 or logits.numel() != labels.numel():
        raise ValueError("metrics require equal non-empty logits and labels")
    positives = int((labels > 0.5).sum().item())
    negatives = int((labels <= 0.5).sum().item())
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC/AP require both positive and negative examples")

    probability = logits.sigmoid()
    prediction = probability >= 0.5
    truth = labels > 0.5
    tp = int((prediction & truth).sum().item())
    fp = int((prediction & ~truth).sum().item())
    tn = int((~prediction & ~truth).sum().item())
    fn = int((~prediction & truth).sum().item())

    order = torch.argsort(probability, descending=True, stable=True)
    sorted_truth = truth[order].to(torch.float64)
    cumulative_tp = sorted_truth.cumsum(0)
    cumulative_fp = (1.0 - sorted_truth).cumsum(0)
    distinct = torch.ones_like(sorted_truth, dtype=torch.bool)
    distinct[:-1] = probability[order][:-1] != probability[order][1:]
    tpr = torch.cat((torch.zeros(1), cumulative_tp[distinct] / positives))
    fpr_curve = torch.cat((torch.zeros(1), cumulative_fp[distinct] / negatives))
    auroc = torch.trapz(tpr, fpr_curve).item()
    precision_curve = cumulative_tp / torch.arange(1, labels.numel() + 1, dtype=torch.float64)
    average_precision = (precision_curve * sorted_truth).sum().item() / positives

    return {
        "count": float(labels.numel()),
        "prevalence": positives / labels.numel(),
        "bce": torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item(),
        "brier": (probability - labels).square().mean().item(),
        "auroc": auroc,
        "average_precision": average_precision,
        "accuracy_at_0_5": (prediction == truth).to(torch.float64).mean().item(),
        "recall_at_0_5": tp / positives,
        "precision_at_0_5": tp / max(tp + fp, 1),
        "false_positive_rate_at_0_5": fp / negatives,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


@torch.no_grad()
def evaluate_risk_model(
    model: CorrectiveTakeoverRiskMLP,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logits: list[Tensor] = []
    labels: list[Tensor] = []
    categories: list[str] = []
    for row in rows:
        if float(torch.as_tensor(row["label_mask"]).item()) <= 0.5:
            continue
        state = torch.as_tensor(row["state_vec"], dtype=torch.float32, device=device).unsqueeze(0)
        action = torch.as_tensor(row["action_chunk"], dtype=torch.float32, device=device).unsqueeze(0)
        logits.append(model(state, action).cpu())
        labels.append(torch.as_tensor(row["label"], dtype=torch.float32).reshape(1))
        categories.append(str(row["category"]))
    all_logits = torch.cat(logits)
    all_labels = torch.cat(labels)
    result: dict[str, Any] = {"overall": binary_classification_metrics(all_logits, all_labels)}
    indices_by_category: dict[str, list[int]] = defaultdict(list)
    for index, category in enumerate(categories):
        indices_by_category[category].append(index)
    per_category: dict[str, Any] = {}
    for category, indices in sorted(indices_by_category.items()):
        selected_logits = all_logits[indices]
        selected_labels = all_labels[indices]
        positives = int((selected_labels > 0.5).sum().item())
        negatives = int((selected_labels <= 0.5).sum().item())
        per_category[category] = (
            binary_classification_metrics(selected_logits, selected_labels)
            if positives and negatives
            else {
                "count": len(indices),
                "positive": positives,
                "negative": negatives,
                "auroc": None,
                "average_precision": None,
            }
        )
    result["per_category"] = per_category
    return result


def train_corrective_risk(
    *,
    dataset_dir: Path,
    output_checkpoint: Path,
    report_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid risk training hyperparameters")
    metadata_path = dataset_dir / "metadata.json"
    train_path = dataset_dir / "actor_trust_train.pt"
    val_path = dataset_dir / "actor_trust_val.pt"
    for path in (metadata_path, train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(f"risk dataset artifact not found: {path}")
    metadata = json.loads(metadata_path.read_text())
    train_rows = torch.load(train_path, map_location="cpu", weights_only=False)
    val_rows = torch.load(val_path, map_location="cpu", weights_only=False)
    train_positive, train_negative, train_masked = _require_two_classes(train_rows, "train")
    val_positive, val_negative, val_masked = _require_two_classes(val_rows, "val")
    pos_weight = train_negative / train_positive

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    run_device = torch.device(device)
    model = CorrectiveTakeoverRiskMLP(
        state_dim=int(metadata["dimensions"]["state_dim"]),
        action_dim=int(metadata["dimensions"]["action_flat_dim"]),
    ).to(run_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        _RiskDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        numerator = denominator = 0.0
        for batch in loader:
            mask = batch["mask"].to(run_device)
            if not bool((mask > 0.5).any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["state"].to(run_device), batch["action"].to(run_device))
            loss = masked_risk_bce_with_logits(
                logits,
                batch["label"].to(run_device),
                mask,
                pos_weight=pos_weight,
            )
            loss.backward()
            optimizer.step()
            eligible = float((mask > 0.5).sum().item())
            numerator += float(loss.detach().item()) * eligible
            denominator += eligible
        history.append({"epoch": float(epoch + 1), "train_masked_bce": numerator / denominator})

    checkpoint_metadata = {
        "dataset": {
            "directory": str(dataset_dir.expanduser().resolve()),
            "metadata_sha256": _sha256_file(metadata_path),
            "train_sha256": _sha256_file(train_path),
            "val_sha256": _sha256_file(val_path),
            "train_episode_uids": metadata["episodes"]["train_uids"],
            "val_episode_uids": metadata["episodes"]["val_uids"],
        },
        "label_semantics": metadata["semantics"],
        "normalization": "none",
        "class_counts": {
            "train": {"positive": train_positive, "negative": train_negative, "masked": train_masked},
            "val": {"positive": val_positive, "negative": val_negative, "masked": val_masked},
        },
        "pos_weight": pos_weight,
        "optimizer": "AdamW",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "action_semantics": metadata["semantics"]["risk_action_semantics"],
        "primary_future_k": metadata["semantics"]["primary_future_k"],
        "primary_future_k_anchor_horizon": metadata["semantics"].get(
            "primary_future_k_anchor_horizon",
            metadata["semantics"]["primary_future_k"],
        ),
        "anchor_stride_frames": metadata["semantics"].get(
            "anchor_stride_frames",
            metadata["semantics"]["frame_stride"],
        ),
        "anchor_horizon_interpretation": (
            "short pre-takeover horizon over cache anchors; not executed action chunks"
        ),
        "frame_stride": metadata["semantics"]["frame_stride"],
        "fps": metadata["semantics"]["fps"],
    }
    save_corrective_risk_checkpoint(output_checkpoint, model, checkpoint_metadata)
    frozen_model, loaded_metadata = load_corrective_risk_checkpoint(
        output_checkpoint,
        map_location=run_device,
        freeze=True,
    )
    frozen_model.to(run_device)
    report = {
        "checkpoint": str(output_checkpoint.expanduser().resolve()),
        "checkpoint_metadata": loaded_metadata,
        "history": history,
        "train_metrics": evaluate_risk_model(frozen_model, train_rows, device=run_device),
        "val_metrics": evaluate_risk_model(frozen_model, val_rows, device=run_device),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_corrective_risk(
        dataset_dir=args.dataset_dir,
        output_checkpoint=args.output_checkpoint,
        report_path=args.report_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
