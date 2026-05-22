"""
evaluation.py
=============
Cortex-RAG: Evaluation and Explainability

Metrics:
  - Per-task AUROC, AUPRC, F1, Precision, Recall, Accuracy
  - Calibration (Brier Score, Expected Calibration Error)
  - Retrieval quality (faithfulness proxy)
  - Attention-based feature importance

Explainability:
  - Attention weight visualisation
  - Integrated Gradients (approximated)
  - SHAP-like feature importance
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

TASK_NAMES = ["readmission_30d", "adverse_event_90d", "mortality_1yr"]


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Union[float, Dict[str, float]] = 0.5,
) -> Dict:
    """
    Compute per-task and aggregate evaluation metrics.

    Args:
        y_true: (n_samples, n_tasks) binary labels
        y_prob: (n_samples, n_tasks) predicted probabilities
        threshold: decision threshold for binary classification metrics
    """
    metrics = {}
    aurocs, auprcs, briers = [], [], []

    for i, task in enumerate(TASK_NAMES):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        task_threshold = threshold.get(task, 0.5) if isinstance(threshold, dict) else threshold
        yhat = (yp >= task_threshold).astype(int)

        # Guard against degenerate folds
        if len(np.unique(yt)) < 2:
            logger.warning(f"Task {task}: only one class present, skipping AUC.")
            continue

        auroc = roc_auc_score(yt, yp)
        auprc = average_precision_score(yt, yp)
        brier = brier_score_loss(yt, yp)
        acc = accuracy_score(yt, yhat)
        f1 = f1_score(yt, yhat, zero_division=0)
        prec = precision_score(yt, yhat, zero_division=0)
        rec = recall_score(yt, yhat, zero_division=0)

        metrics[f"{task}_auroc"] = round(auroc, 4)
        metrics[f"{task}_auprc"] = round(auprc, 4)
        metrics[f"{task}_brier"] = round(brier, 4)
        metrics[f"{task}_accuracy"] = round(acc, 4)
        metrics[f"{task}_f1"] = round(f1, 4)
        metrics[f"{task}_precision"] = round(prec, 4)
        metrics[f"{task}_recall"] = round(rec, 4)
        metrics[f"{task}_threshold"] = round(float(task_threshold), 4)

        aurocs.append(auroc)
        auprcs.append(auprc)
        briers.append(brier)

    if aurocs:
        metrics["mean_auroc"] = round(np.mean(aurocs), 4)
        metrics["mean_auprc"] = round(np.mean(auprcs), 4)
        metrics["mean_brier"] = round(np.mean(briers), 4)

    return metrics


def tune_task_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.40,
) -> Dict[str, float]:
    """
    Tune per-task thresholds to maximize F1 under a precision floor.
    Falls back to the best F1 threshold if no threshold satisfies the floor.
    """
    thresholds: Dict[str, float] = {}
    grid = np.linspace(0.01, 0.99, 99)

    for i, task in enumerate(TASK_NAMES):
        yt = y_true[:, i]
        yp = y_prob[:, i]

        if len(np.unique(yt)) < 2:
            thresholds[task] = 0.5
            continue

        best_thr = 0.5
        best_f1 = -1.0
        best_f1_thr = 0.5
        best_f1_with_precision = -1.0
        best_f1_with_precision_thr = 0.5

        for thr in grid:
            yhat = (yp >= thr).astype(int)
            prec = precision_score(yt, yhat, zero_division=0)
            rec = recall_score(yt, yhat, zero_division=0)
            f1 = f1_score(yt, yhat, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_f1_thr = float(thr)

            if prec >= min_precision and f1 > best_f1_with_precision:
                best_f1_with_precision = f1
                best_f1_with_precision_thr = float(thr)

        thresholds[task] = (
            best_f1_with_precision_thr
            if best_f1_with_precision >= 0
            else best_f1_thr
        )

    return thresholds


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) for a single task.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        # Last bin should include probability = 1.0 exactly
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return round(ece / len(y_true), 4)


# ---------------------------------------------------------------------------
# Retrieval Quality
# ---------------------------------------------------------------------------

def retrieval_faithfulness_score(
    retrieved_contexts: List[Dict],
    ground_truth_conditions: List[str],
) -> float:
    """
    Proxy metric: fraction of ground-truth conditions present
    in the retrieved context text.
    (In production, use BERTScore or an NLI model for faithfulness.)
    """
    if not ground_truth_conditions:
        return 0.0
    hits = 0
    for ctx in retrieved_contexts:
        ctx_text = ctx.get("context", "").lower()
        for cond in ground_truth_conditions:
            if cond.lower() in ctx_text:
                hits += 1
    return round(hits / (len(ground_truth_conditions) * max(len(retrieved_contexts), 1)), 4)


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

class AttentionExplainer:
    """
    Extracts and interprets attention weights from the RAG model
    to surface which retrieved patients most influenced the prediction.
    """

    def explain_prediction(
        self,
        prediction: Dict,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Returns an explanation dict with ranked retrieved patient influences.
        """
        retrieved = prediction.get("retrieved_similar_patients", [])
        attn_items = sorted(retrieved, key=lambda x: x["attention_weight"], reverse=True)
        return {
            "patient_id": prediction["patient_id"],
            "top_influential_contexts": attn_items,
            "risk_score": prediction["risk_score"],
            "explanation_text": self._build_explanation_text(prediction, attn_items),
        }

    def _build_explanation_text(self, prediction: Dict, attn_items: List[Dict]) -> str:
        lines = [
            f"Prediction Explanation for {prediction['patient_id']}:",
            f"  Composite Risk Score: {prediction['risk_score']:.2%}",
            "",
            "Most influential retrieved patients:",
        ]
        for item in attn_items:
            lines.append(
                f"  - {item['patient_id']} "
                f"(attention weight: {item['attention_weight']:.3f})"
            )
        lines += [
            "",
            "Task-specific probabilities:",
            f"  • 30-day Readmission:  {prediction['readmission_30d_prob']:.1%}",
            f"  • 90-day Adverse:      {prediction['adverse_event_90d_prob']:.1%}",
            f"  • 1-year Mortality:    {prediction['mortality_1yr_prob']:.1%}",
        ]
        return "\n".join(lines)


class GradientFeatureImportance:
    """
    Approximated integrated gradients for structured feature importance.
    Identifies which clinical features most influenced the prediction.
    """

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def compute_importance(
        self,
        structured_features: np.ndarray,
        graph_embedding: np.ndarray,
        context_embeddings: np.ndarray,
        target_task: int = 0,
        n_steps: int = 20,
    ) -> np.ndarray:
        """
        Integrated gradients approximation.
        Returns importance scores of shape (structured_dim,).
        """
        import torch

        self.model.eval()
        baseline = np.zeros_like(structured_features)

        sf_input = torch.tensor(structured_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        ge_input = torch.tensor(graph_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        ctx_input = torch.tensor(context_embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
        baseline_t = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).to(self.device)

        gradients = []
        for alpha in np.linspace(0, 1, n_steps):
            interp = baseline_t + alpha * (sf_input - baseline_t)
            interp.requires_grad_(True)
            outputs = self.model(interp, ge_input, ctx_input)
            logit = outputs["logits"][0, target_task]
            logit.backward()
            gradients.append(interp.grad.detach().cpu().numpy()[0])

        avg_grads = np.mean(gradients, axis=0)
        integrated_grads = (structured_features - baseline) * avg_grads
        return integrated_grads


# ---------------------------------------------------------------------------
# Evaluation Report
# ---------------------------------------------------------------------------

def generate_evaluation_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: List[str],
    thresholds: Optional[Dict[str, float]] = None,
    predictions: Optional[List[Dict]] = None,
) -> str:
    """
    Generate a comprehensive plain-text evaluation report.
    """
    metrics = compute_metrics(y_true, y_prob, threshold=thresholds or 0.5)
    lines = [
        "=" * 60,
        "CORTEX-RAG EVALUATION REPORT",
        "=" * 60,
        f"Samples evaluated: {len(y_true)}",
        "",
        "PER-TASK PERFORMANCE",
        "-" * 40,
    ]
    for task in TASK_NAMES:
        lines.append(f"\nTask: {task.upper()}")
        for m in ["auroc", "auprc", "f1", "precision", "recall", "accuracy", "brier"]:
            key = f"{task}_{m}"
            if key in metrics:
                lines.append(f"  {m:<12}: {metrics[key]:.4f}")
        if f"{task}_threshold" in metrics:
            lines.append(f"  {'threshold':<12}: {metrics[f'{task}_threshold']:.4f}")

    lines += [
        "",
        "AGGREGATE",
        "-" * 40,
        f"  Mean AUROC : {metrics.get('mean_auroc', 0):.4f}",
        f"  Mean AUPRC : {metrics.get('mean_auprc', 0):.4f}",
        f"  Mean Brier : {metrics.get('mean_brier', 0):.4f}",
    ]

    # Calibration per task
    lines += ["", "CALIBRATION (ECE)", "-" * 40]
    for i, task in enumerate(TASK_NAMES):
        if len(np.unique(y_true[:, i])) > 1:
            ece = expected_calibration_error(y_true[:, i], y_prob[:, i])
            lines.append(f"  {task:<30}: ECE = {ece:.4f}")

    lines += ["", "=" * 60]
    return "\n".join(lines)


def save_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    output_dir: str = "outputs/evaluation",
) -> List[str]:
    """
    Save ROC, Precision-Recall, and confusion matrix plots per task.
    Returns list of generated file paths.
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    saved_files: List[str] = []
    thresholds = thresholds or {t: 0.5 for t in TASK_NAMES}

    for i, task in enumerate(TASK_NAMES):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if len(np.unique(yt)) < 2:
            continue

        thr = float(thresholds.get(task, 0.5))
        yhat = (yp >= thr).astype(int)

        # ROC + PR in one figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fpr, tpr, _ = roc_curve(yt, yp)
        prec_curve, rec_curve, _ = precision_recall_curve(yt, yp)
        auroc = roc_auc_score(yt, yp)
        auprc = average_precision_score(yt, yp)

        axes[0].plot(fpr, tpr, label=f"AUROC={auroc:.3f}", color="#0b6e4f")
        axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        axes[0].set_title(f"ROC - {task}")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend(loc="lower right")

        axes[1].plot(rec_curve, prec_curve, label=f"AUPRC={auprc:.3f}", color="#1f77b4")
        axes[1].set_title(f"Precision-Recall - {task}")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(loc="lower left")

        fig.tight_layout()
        curve_file = os.path.join(output_dir, f"{task}_roc_pr.png")
        fig.savefig(curve_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(curve_file)

        # Confusion matrix at tuned threshold
        cm = confusion_matrix(yt, yhat)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5))
        im = ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_title(f"Confusion Matrix - {task} @ {thr:.2f}")
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Neg", "Pos"])
        ax_cm.set_yticklabels(["Neg", "Pos"])

        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax_cm.text(c, r, int(cm[r, c]), ha="center", va="center", color="black")

        fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
        fig_cm.tight_layout()
        cm_file = os.path.join(output_dir, f"{task}_confusion_matrix.png")
        fig_cm.savefig(cm_file, dpi=150, bbox_inches="tight")
        plt.close(fig_cm)
        saved_files.append(cm_file)

    return saved_files
