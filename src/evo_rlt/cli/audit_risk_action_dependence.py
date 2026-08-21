"""Matched Full/State-only/Shuffled-action corrective-risk ablation."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from evo_rlt.cli.train_corrective_risk import (
    _RiskDataset,
    _label_counts,
    _require_two_classes,
    _sha256_file,
    binary_classification_metrics,
)
from evo_rlt.core.corrective_risk import (
    CorrectiveTakeoverRiskMLP,
    masked_risk_bce_with_logits,
    save_corrective_risk_checkpoint,
)


FULL = "full_state_action"
STATE_ONLY = "state_only"
SHUFFLED = "shuffled_action"
VARIANTS = (FULL, STATE_ONLY, SHUFFLED)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_variant(
    model: CorrectiveTakeoverRiskMLP,
    state: Tensor,
    action: Tensor,
) -> Tensor:
    return model(state) if model.action_dim == 0 else model(state, action)


def _deranged_permutation(length: int, *, seed: int) -> list[int]:
    if length < 2:
        raise ValueError("shuffled-action ablation requires at least two rows per split")
    generator = torch.Generator().manual_seed(seed)
    donors = torch.randperm(length, generator=generator)
    identity = torch.arange(length)
    for _ in range(length):
        if not bool((donors == identity).any()):
            return donors.tolist()
        donors = donors.roll(1)
    raise RuntimeError("could not construct deterministic deranged action permutation")


def make_shuffled_action_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle actions inside one split while retaining each recipient's label/state."""
    donor_indices = _deranged_permutation(len(rows), seed=seed)
    shuffled: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for recipient_index, donor_index in enumerate(donor_indices):
        recipient = copy.copy(rows[recipient_index])
        recipient["action_chunk"] = torch.as_tensor(
            rows[donor_index]["action_chunk"]
        ).detach().clone()
        shuffled.append(recipient)
        mapping.append(
            {
                "recipient_index": recipient_index,
                "donor_index": donor_index,
                "recipient_episode_uid": str(rows[recipient_index]["episode_uid"]),
                "donor_episode_uid": str(rows[donor_index]["episode_uid"]),
            }
        )
    return shuffled, mapping


def _quantiles(values: Tensor) -> dict[str, float | int | None]:
    values = values.detach().cpu().to(torch.float64).reshape(-1)
    if not values.numel():
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    p10, p50, p90 = torch.quantile(
        values, torch.tensor([0.1, 0.5, 0.9], dtype=values.dtype)
    ).tolist()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
    }


@torch.no_grad()
def _predict_eligible(
    model: CorrectiveTakeoverRiskMLP,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    eligible_rows = [
        row for row in rows if float(torch.as_tensor(row["label_mask"]).item()) > 0.5
    ]
    loader = DataLoader(_RiskDataset(eligible_rows), batch_size=batch_size, shuffle=False)
    logits: list[Tensor] = []
    labels: list[Tensor] = []
    model.eval()
    for batch in loader:
        logits.append(
            _forward_variant(
                model,
                batch["state"].to(device),
                batch["action"].to(device),
            ).cpu()
        )
        labels.append(batch["label"].cpu())
    return torch.cat(logits), torch.cat(labels), eligible_rows


def _semantic_score_report(
    logits: Tensor,
    labels: Tensor,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    probability = logits.sigmoid()
    result: dict[str, Any] = {
        "overall": binary_classification_metrics(logits, labels),
        "confusion_matrix_at_0_5": {
            "tp": int(((probability >= 0.5) & (labels > 0.5)).sum().item()),
            "fp": int(((probability >= 0.5) & (labels <= 0.5)).sum().item()),
            "tn": int(((probability < 0.5) & (labels <= 0.5)).sum().item()),
            "fn": int(((probability < 0.5) & (labels > 0.5)).sum().item()),
        },
    }

    per_category: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if str(row["category"]) == category]
        category_labels = labels[indices]
        category_logits = logits[indices]
        positives = int((category_labels > 0.5).sum().item())
        negatives = int((category_labels <= 0.5).sum().item())
        per_category[category] = {
            "count": len(indices),
            "positive": positives,
            "negative": negatives,
            "score_distribution": _quantiles(category_logits.sigmoid()),
        }
        if positives and negatives:
            per_category[category]["metrics"] = binary_classification_metrics(
                category_logits, category_labels
            )
    result["per_category"] = per_category

    corrective_by_episode: dict[str, list[int]] = defaultdict(list)
    success_indices: list[int] = []
    success_by_episode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        episode_uid = str(row["episode_uid"])
        category = str(row["category"])
        if category == "corrective":
            corrective_by_episode[episode_uid].append(index)
        elif category == "autonomous_success":
            success_indices.append(index)
            success_by_episode[episode_uid].append(index)

    event_rows: list[dict[str, Any]] = []
    for episode_uid, indices in sorted(corrective_by_episode.items()):
        positive_indices = [index for index in indices if labels[index] > 0.5]
        ordered = sorted(indices, key=lambda index: int(rows[index].get("anchor_index", index)))
        scores = probability[indices]
        positive_scores = probability[positive_indices]
        if not positive_scores.numel():
            raise ValueError(f"corrective episode has no positive anchors: {episode_uid}")
        event_rows.append(
            {
                "episode_uid": episode_uid,
                "eligible_anchor_count": len(indices),
                "positive_anchor_count": len(positive_indices),
                "mean_prefix_score": float(scores.mean().item()),
                "max_prefix_score": float(scores.max().item()),
                "mean_positive_anchor_score": float(positive_scores.mean().item()),
                "max_positive_anchor_score": float(positive_scores.max().item()),
                "last_anchor_score": float(probability[ordered[-1]].item()),
                "detected_any_positive_at_0_5": bool((positive_scores >= 0.5).any()),
            }
        )
    result["corrective_event_level"] = {
        "events": len(event_rows),
        "event_detection_recall_at_0_5": (
            sum(row["detected_any_positive_at_0_5"] for row in event_rows) / len(event_rows)
            if event_rows
            else None
        ),
        "mean_positive_anchor_score": _quantiles(
            torch.tensor([row["mean_positive_anchor_score"] for row in event_rows])
        ),
        "last_anchor_score": _quantiles(
            torch.tensor([row["last_anchor_score"] for row in event_rows])
        ),
        "per_event": event_rows,
    }
    result["autonomous_success_score_distribution"] = {
        "anchor_scores": _quantiles(probability[success_indices]),
        "episode_mean_scores": _quantiles(
            torch.tensor(
                [probability[indices].mean().item() for indices in success_by_episode.values()]
            )
        ),
        "episode_max_scores": _quantiles(
            torch.tensor(
                [probability[indices].max().item() for indices in success_by_episode.values()]
            )
        ),
    }
    return result


def _train_variant(
    *,
    variant: str,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    state_dim: int,
    action_dim: int,
    pos_weight: float,
    output_dir: Path,
    checkpoint_metadata: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], Tensor, list[dict[str, Any]]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown risk ablation variant: {variant}")
    _seed_everything(seed)
    model = CorrectiveTakeoverRiskMLP(
        state_dim=state_dim,
        action_dim=0 if variant == STATE_ONLY else action_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        _RiskDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, float]] = []
    optimizer_steps = 0
    for epoch in range(epochs):
        model.train()
        numerator = denominator = 0.0
        for batch in loader:
            mask = batch["mask"].to(device)
            if not bool((mask > 0.5).any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_variant(
                model,
                batch["state"].to(device),
                batch["action"].to(device),
            )
            loss = masked_risk_bce_with_logits(
                logits,
                batch["label"].to(device),
                mask,
                pos_weight=pos_weight,
            )
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            eligible = float((mask > 0.5).sum().item())
            numerator += float(loss.detach().item()) * eligible
            denominator += eligible
        history.append({"epoch": epoch + 1, "train_masked_bce": numerator / denominator})

    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = variant_dir / "risk.pt"
    metadata = {
        **checkpoint_metadata,
        "ablation_variant": variant,
        "optimizer_steps": optimizer_steps,
    }
    save_corrective_risk_checkpoint(checkpoint_path, model, metadata)
    train_logits, train_labels, train_eligible = _predict_eligible(
        model, train_rows, device=device, batch_size=batch_size
    )
    val_logits, val_labels, val_eligible = _predict_eligible(
        model, val_rows, device=device, batch_size=batch_size
    )
    report = {
        "variant": variant,
        "checkpoint": str(checkpoint_path.resolve()),
        "model_input": "state_vec" if variant == STATE_ONLY else "state_vec + action_chunk",
        "history": history,
        "optimizer_steps": optimizer_steps,
        "train": _semantic_score_report(train_logits, train_labels, train_eligible),
        "validation": _semantic_score_report(val_logits, val_labels, val_eligible),
    }
    return report, val_logits, val_eligible


def _delta_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"valid_replicates": 0, "median": None, "ci95": None, "p_gt_zero": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    low, median, high = torch.quantile(
        tensor, torch.tensor([0.025, 0.5, 0.975], dtype=tensor.dtype)
    ).tolist()
    return {
        "valid_replicates": len(values),
        "median": median,
        "ci95": [low, high],
        "p_gt_zero": float((tensor > 0).to(torch.float64).mean().item()),
    }


def paired_episode_bootstrap(
    logits_by_variant: dict[str, Tensor],
    labels: Tensor,
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired episode bootstrap of Full-minus-ablation AP/AUROC."""
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    episode_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        episode_indices[str(row["episode_uid"])].append(index)
    episode_uids = sorted(episode_indices)
    if len(episode_uids) < 2:
        raise ValueError("episode bootstrap requires at least two validation episodes")
    rng = random.Random(seed)
    deltas: dict[str, dict[str, list[float]]] = {
        STATE_ONLY: {"average_precision": [], "auroc": []},
        SHUFFLED: {"average_precision": [], "auroc": []},
    }
    skipped_single_class = 0
    for _ in range(replicates):
        sampled_uids = [rng.choice(episode_uids) for _ in episode_uids]
        indices = [index for uid in sampled_uids for index in episode_indices[uid]]
        sampled_labels = labels[indices]
        if not bool((sampled_labels > 0.5).any()) or not bool((sampled_labels <= 0.5).any()):
            skipped_single_class += 1
            continue
        metrics = {
            variant: binary_classification_metrics(logits[indices], sampled_labels)
            for variant, logits in logits_by_variant.items()
        }
        for ablation in (STATE_ONLY, SHUFFLED):
            for metric in ("average_precision", "auroc"):
                deltas[ablation][metric].append(
                    metrics[FULL][metric] - metrics[ablation][metric]
                )
    return {
        "unit": "episode",
        "requested_replicates": replicates,
        "skipped_single_class_replicates": skipped_single_class,
        "full_minus_state_only": {
            metric: _delta_summary(values) for metric, values in deltas[STATE_ONLY].items()
        },
        "full_minus_shuffled_action": {
            metric: _delta_summary(values) for metric, values in deltas[SHUFFLED].items()
        },
    }


def _action_signal_verdict(
    point_deltas: dict[str, dict[str, float]],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    state_ap = bootstrap["full_minus_state_only"]["average_precision"]
    shuffled_ap = bootstrap["full_minus_shuffled_action"]["average_precision"]
    state_auc = bootstrap["full_minus_state_only"]["auroc"]
    shuffled_auc = bootstrap["full_minus_shuffled_action"]["auroc"]
    ap_points_positive = (
        point_deltas["full_minus_state_only"]["average_precision"] > 0
        and point_deltas["full_minus_shuffled_action"]["average_precision"] > 0
    )
    auc_points_nonnegative = (
        point_deltas["full_minus_state_only"]["auroc"] >= 0
        and point_deltas["full_minus_shuffled_action"]["auroc"] >= 0
    )
    min_ap_probability = min(state_ap["p_gt_zero"], shuffled_ap["p_gt_zero"])
    min_auc_probability = min(state_auc["p_gt_zero"], shuffled_auc["p_gt_zero"])
    if ap_points_positive and auc_points_nonnegative and min_ap_probability >= 0.8 and min_auc_probability >= 0.65:
        verdict = "ACTION_SIGNAL_PASS"
    elif ap_points_positive and auc_points_nonnegative and min_ap_probability >= 0.65:
        verdict = "ACTION_SIGNAL_WEAK_PASS"
    else:
        verdict = "ACTION_SIGNAL_FAIL"
    return {
        "verdict": verdict,
        "preregistered_rule": {
            "pass": (
                "both point AUPRC deltas >0, both point AUROC deltas >=0, "
                "min P(delta AUPRC>0)>=0.8, min P(delta AUROC>0)>=0.65"
            ),
            "weak_pass": (
                "both point AUPRC deltas >0, both point AUROC deltas >=0, "
                "min P(delta AUPRC>0)>=0.65"
            ),
            "fail": "otherwise",
        },
        "phase_2_authorized": verdict in {"ACTION_SIGNAL_PASS", "ACTION_SIGNAL_WEAK_PASS"},
        "interpretation": (
            "action contributes predictive information"
            if verdict != "ACTION_SIGNAL_FAIL"
            else "risk primarily behaves as a state/progress intervention predictor"
        ),
    }


def run_action_dependence_ablation(
    *,
    dataset_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    bootstrap_replicates: int,
    device_name: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid risk ablation hyperparameters")
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
    val_positive, val_negative, val_masked = _require_two_classes(val_rows, "validation")
    train_uids = {str(row["episode_uid"]) for row in train_rows}
    val_uids = {str(row["episode_uid"]) for row in val_rows}
    if train_uids & val_uids:
        raise ValueError("episode leakage between risk train and validation splits")

    shuffled_train, train_mapping = make_shuffled_action_rows(train_rows, seed=seed + 10_001)
    shuffled_val, val_mapping = make_shuffled_action_rows(val_rows, seed=seed + 20_001)
    output_dir.mkdir(parents=True, exist_ok=False)
    semantics = metadata["semantics"]
    primary_k = int(
        semantics.get("primary_future_k_anchor_horizon", semantics["primary_future_k"])
    )
    anchor_stride = int(semantics.get("anchor_stride_frames", semantics["frame_stride"]))
    fps = float(semantics["fps"])
    horizon_seconds = primary_k * anchor_stride / fps
    common_metadata = {
        "dataset": {
            "directory": str(dataset_dir.resolve()),
            "metadata_sha256": _sha256_file(metadata_path),
            "train_sha256": _sha256_file(train_path),
            "val_sha256": _sha256_file(val_path),
            "train_episode_uids": sorted(train_uids),
            "val_episode_uids": sorted(val_uids),
        },
        "label_semantics": semantics,
        "primary_future_k": primary_k,
        "primary_future_k_anchor_horizon": primary_k,
        "anchor_stride_frames": anchor_stride,
        "fps": fps,
        "anchor_horizon_seconds": horizon_seconds,
        "anchor_horizon_interpretation": (
            "short pre-takeover horizon over cache anchors; not executed action chunks"
        ),
        "normalization": "none",
        "pos_weight": train_negative / train_positive,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
    }
    variants: dict[str, Any] = {}
    logits_by_variant: dict[str, Tensor] = {}
    eligible_rows: list[dict[str, Any]] | None = None
    inputs = {
        FULL: (train_rows, val_rows),
        STATE_ONLY: (train_rows, val_rows),
        SHUFFLED: (shuffled_train, shuffled_val),
    }
    optimizer_steps: set[int] = set()
    for variant in VARIANTS:
        report, logits, variant_eligible = _train_variant(
            variant=variant,
            train_rows=inputs[variant][0],
            val_rows=inputs[variant][1],
            state_dim=int(metadata["dimensions"]["state_dim"]),
            action_dim=int(metadata["dimensions"]["action_flat_dim"]),
            pos_weight=train_negative / train_positive,
            output_dir=output_dir,
            checkpoint_metadata=common_metadata,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            device=torch.device(device_name),
        )
        variants[variant] = report
        logits_by_variant[variant] = logits
        optimizer_steps.add(int(report["optimizer_steps"]))
        if eligible_rows is None:
            eligible_rows = variant_eligible
    if len(optimizer_steps) != 1:
        raise AssertionError(f"risk variants used different optimizer step counts: {optimizer_steps}")
    assert eligible_rows is not None
    labels = torch.tensor(
        [float(torch.as_tensor(row["label"]).item()) for row in eligible_rows]
    )
    point_deltas = {}
    full_metrics = variants[FULL]["validation"]["overall"]
    for ablation, key in ((STATE_ONLY, "full_minus_state_only"), (SHUFFLED, "full_minus_shuffled_action")):
        ablation_metrics = variants[ablation]["validation"]["overall"]
        point_deltas[key] = {
            "average_precision": full_metrics["average_precision"] - ablation_metrics["average_precision"],
            "auroc": full_metrics["auroc"] - ablation_metrics["auroc"],
        }
    bootstrap = paired_episode_bootstrap(
        logits_by_variant,
        labels,
        eligible_rows,
        replicates=bootstrap_replicates,
        seed=seed + 30_001,
    )
    decision = _action_signal_verdict(point_deltas, bootstrap)
    report = {
        "schema_version": 1,
        "status": "COMPLETE",
        "experiment": "matched corrective-risk action-dependence ablation",
        "dataset": common_metadata["dataset"],
        "matched_controls": {
            "same_episode_split": True,
            "same_labels": True,
            "same_seed": seed,
            "same_epochs": epochs,
            "same_optimizer_steps": optimizer_steps.pop(),
            "same_hidden_dims": [256, 128],
            "same_loss": "masked weighted BCEWithLogits",
            "same_pos_weight": train_negative / train_positive,
            "same_batch_order": True,
            "early_stopping": "none; fixed epochs",
        },
        "class_counts": {
            "train": {"positive": train_positive, "negative": train_negative, "masked": train_masked},
            "validation": {"positive": val_positive, "negative": val_negative, "masked": val_masked},
        },
        "anchor_horizon": {
            "primary_future_k_anchor_horizon": primary_k,
            "anchor_stride_frames": anchor_stride,
            "fps": fps,
            "horizon_seconds": horizon_seconds,
            "formula": "K_anchor * anchor_stride_frames / fps",
            "interpretation": "short pre-takeover horizon; not three executed action chunks",
        },
        "variants": variants,
        "shuffled_action_mapping": {
            "train_seed": seed + 10_001,
            "validation_seed": seed + 20_001,
            "train": train_mapping,
            "validation": val_mapping,
            "cross_split_shuffle": False,
        },
        "point_deltas": point_deltas,
        "episode_bootstrap": bootstrap,
        "decision": decision,
        "phase_2_gate": (
            "train exactly one proposed C only when phase_2_authorized=true"
        ),
    }
    report_path = output_dir / "action_dependence_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_action_dependence_ablation(
        dataset_dir=args.dataset_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        device_name=args.device,
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    print(f"report={args.output_dir.expanduser().resolve() / 'action_dependence_report.json'}")


if __name__ == "__main__":
    main()
