"""
training.py
===========
Cortex-RAG: Model Training and Optimisation

Implements:
  - ClinicalDataset  (PyTorch Dataset for patient records)
  - RAGTrainer        (training loop, LR scheduling, gradient clipping)
  - Cross-validation utility
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClinicalDataset(Dataset):
    """
    PyTorch Dataset wrapping patient feature matrix and outcome labels.

    Each sample returns:
        structured_features : (structured_dim,)
        graph_embedding     : (graph_emb_dim,)
        context_embeddings  : (top_k, latent_dim)
        labels              : (n_tasks,)
    """

    def __init__(
        self,
        patient_features: np.ndarray,
        graph_embeddings: np.ndarray,
        context_embeddings: np.ndarray,   # (n_patients, top_k, latent_dim)
        labels: np.ndarray,
    ):
        assert len(patient_features) == len(labels), "Feature/label length mismatch"
        self.patient_features = torch.tensor(patient_features, dtype=torch.float32)
        self.graph_embeddings = torch.tensor(graph_embeddings, dtype=torch.float32)
        self.context_embeddings = torch.tensor(context_embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.patient_features[idx],
            self.graph_embeddings[idx],
            self.context_embeddings[idx],
            self.labels[idx],
        )


# ---------------------------------------------------------------------------
# Multi-task Loss
# ---------------------------------------------------------------------------

class MultiTaskClinicalLoss(nn.Module):
    """
    Weighted binary cross-entropy for multi-task clinical prediction.
    Uses aggressive class weighting to focus on rare positive cases.
    Task weights reflect clinical importance (optimized):
      readmission < adverse_event < mortality
    """

    def __init__(
        self,
        task_weights: Optional[List[float]] = None,
        pos_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.task_weights = task_weights or [1.0, 1.8, 2.5]  # Optimized weights
        if pos_weights is None:
            pos_weights = torch.ones(len(self.task_weights), dtype=torch.float32)
        self.register_buffer("pos_weights", pos_weights.float())
        self.mse = nn.MSELoss()

    def forward(
        self,
        logits: torch.Tensor,
        risk_score: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            logits:     (batch, n_tasks)
            risk_score: (batch,)
            labels:     (batch, n_tasks)
        """
        task_losses = []
        for i, w in enumerate(self.task_weights):
            # Standard BCE with aggressive class weighting
            bce_i = nn.functional.binary_cross_entropy_with_logits(
                logits[:, i],
                labels[:, i],
                pos_weight=self.pos_weights[i],
                reduction="mean",
            )
            loss_i = bce_i * w
            task_losses.append(loss_i)

        classification_loss = torch.stack(task_losses).sum()

        # Composite risk score supervision (mean of label tasks)
        composite_label = labels.mean(dim=-1)
        risk_loss = self.mse(torch.sigmoid(risk_score), composite_label) * 0.5

        total_loss = classification_loss + risk_loss

        return total_loss, {
            "total": total_loss.item(),
            "classification": classification_loss.item(),
            "risk_regression": risk_loss.item(),
            "task_losses": [l.item() for l in task_losses],
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RAGTrainer:
    """
    Training orchestrator for the ClinicalRAGModel.
    Supports early stopping, LR scheduling, and gradient clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_labels: Optional[np.ndarray] = None,
        device: str = "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-3,      # Increased L2 regularization
        l1_penalty: float = 1e-5,        # L1 regularization coefficient
        max_grad_norm: float = 1.0,
        patience: int = 10,
        task_weights: Optional[List[float]] = None,
        mixup_alpha: float = 0.2,        # Mixup augmentation strength
    ):
        self.model = model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.l1_penalty = l1_penalty
        self.mixup_alpha = mixup_alpha
        pos_weights = self._compute_pos_weights(train_labels) if train_labels is not None else None
        self.loss_fn = MultiTaskClinicalLoss(pos_weights=pos_weights, task_weights=task_weights)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )
        self.history: Dict[str, List] = {
            "train_loss": [], "val_loss": [],
            "train_auc": [], "val_auc": [],
        }
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._best_state = None

    @staticmethod
    def _compute_pos_weights(train_labels: np.ndarray) -> torch.Tensor:
        """
        Compute per-task positive class weights as neg/pos with clipping.
        """
        pos = np.sum(train_labels == 1, axis=0).astype(np.float32)
        neg = np.sum(train_labels == 0, axis=0).astype(np.float32)
        pos = np.clip(pos, 1.0, None)
        weights = np.clip(neg / pos, 1.0, 50.0)
        return torch.tensor(weights, dtype=torch.float32)

    def _apply_mixup(self, sf, ge, ctx_emb, labels, alpha=0.2):
        """Apply mixup augmentation to batch features only (not labels)."""
        batch_size = sf.size(0)
        if batch_size < 2:
            return sf, ge, ctx_emb, labels
        
        lam = np.random.beta(alpha, alpha)
        indices = torch.randperm(batch_size)
        
        # Mix features only, keep original labels for proper loss computation
        sf_mixed = lam * sf + (1 - lam) * sf[indices]
        ge_mixed = lam * ge + (1 - lam) * ge[indices]
        ctx_emb_mixed = lam * ctx_emb + (1 - lam) * ctx_emb[indices]
        
        return sf_mixed, ge_mixed, ctx_emb_mixed, labels

    def _compute_l1_penalty(self):
        """Compute L1 regularization penalty."""
        l1_penalty = 0
        for param in self.model.parameters():
            l1_penalty += torch.sum(torch.abs(param))
        return l1_penalty

    def _run_epoch(
        self, loader: DataLoader, train: bool = True
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        ctx_mgr = torch.enable_grad() if train else torch.no_grad()
        with ctx_mgr:
            for sf, ge, ctx_emb, labels in loader:
                sf = sf.to(self.device)
                ge = ge.to(self.device)
                ctx_emb = ctx_emb.to(self.device)
                labels = labels.to(self.device)

                # Apply mixup augmentation during training
                if train and self.mixup_alpha > 0:
                    sf, ge, ctx_emb, labels = self._apply_mixup(sf, ge, ctx_emb, labels, self.mixup_alpha)

                outputs = self.model(sf, ge, ctx_emb)
                loss, _ = self.loss_fn(outputs["logits"], outputs["risk_score"], labels)
                
                # Add L1 regularization
                if train and self.l1_penalty > 0:
                    loss = loss + self.l1_penalty * self._compute_l1_penalty()

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                total_loss += loss.item() * len(labels)
                probs = torch.sigmoid(outputs["logits"]).cpu().detach().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        return avg_loss, all_probs, all_labels

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
    ) -> Dict:
        from evaluation import compute_metrics

        logger.info(f"Starting training on {self.device} | epochs={n_epochs}")
        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_probs, tr_labels = self._run_epoch(train_loader, train=True)
            va_loss, va_probs, va_labels = self._run_epoch(val_loader, train=False)
            self.scheduler.step()

            tr_metrics = compute_metrics(tr_labels, tr_probs)
            va_metrics = compute_metrics(va_labels, va_probs)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_auc"].append(tr_metrics.get("mean_auroc", 0))
            self.history["val_auc"].append(va_metrics.get("mean_auroc", 0))

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {tr_loss:.4f} AUC: {tr_metrics.get('mean_auroc', 0):.4f} | "
                    f"Val Loss: {va_loss:.4f} AUC: {va_metrics.get('mean_auroc', 0):.4f}"
                )

            # Early stopping
            if va_loss < self._best_val_loss:
                self._best_val_loss = va_loss
                self._patience_counter = 0
                self._best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        # Restore best weights
        if self._best_state:
            self.model.load_state_dict(self._best_state)
            logger.info("Restored best model weights.")

        return self.history


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    model_factory,
    patient_features: np.ndarray,
    graph_embeddings: np.ndarray,
    context_embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    n_epochs: int = 30,
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict:
    """
    Stratified k-fold cross-validation.
    `model_factory` is a callable that returns a fresh ClinicalRAGModel instance.
    """
    from evaluation import compute_metrics

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Stratify on the first task (readmission)
    strat_labels = labels[:, 0].astype(int)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_features, strat_labels)):
        logger.info(f"--- Fold {fold + 1}/{n_splits} ---")

        train_ds = ClinicalDataset(
            patient_features[train_idx], graph_embeddings[train_idx],
            context_embeddings[train_idx], labels[train_idx]
        )
        val_ds = ClinicalDataset(
            patient_features[val_idx], graph_embeddings[val_idx],
            context_embeddings[val_idx], labels[val_idx]
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = model_factory()
        trainer = RAGTrainer(model, train_labels=labels[train_idx], device=device, patience=8)
        trainer.train(train_loader, val_loader, n_epochs=n_epochs)

        # Evaluate on validation fold
        _, probs, true_labels = trainer._run_epoch(val_loader, train=False)
        metrics = compute_metrics(true_labels, probs)
        fold_metrics.append(metrics)
        logger.info(f"Fold {fold + 1} | AUC: {metrics.get('mean_auroc', 0):.4f}")

    # Aggregate across folds
    aggregated = {}
    all_keys = fold_metrics[0].keys()
    for k in all_keys:
        vals = [m[k] for m in fold_metrics if isinstance(m.get(k), (int, float))]
        if vals:
            aggregated[f"{k}_mean"] = round(np.mean(vals), 4)
            aggregated[f"{k}_std"] = round(np.std(vals), 4)

    logger.info(f"CV Results: {aggregated}")
    return {"fold_metrics": fold_metrics, "aggregated": aggregated}


# ---------------------------------------------------------------------------
# Data preparation helper
# ---------------------------------------------------------------------------

def prepare_training_data(
    patient_features: pd.DataFrame,
    outcomes: pd.DataFrame,
    graph_embeddings_map: Dict,       # patient_id -> np.ndarray (graph_emb_dim,)
    context_embeddings_map: Dict,     # patient_id -> np.ndarray (top_k, latent_dim)
    top_k: int = 3,
    latent_dim: int = 128,
    graph_emb_dim: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Aligns patient features, graph embeddings, context embeddings, and labels
    into numpy arrays ready for DataLoader consumption.
    """
    # Align on patient_id
    merged = patient_features.merge(outcomes, on="patient_id", how="inner")
    patient_ids = merged["patient_id"].tolist()

    # Structured features: drop id + categorical text columns
    drop_cols = ["patient_id", "gender", "ethnicity", "age_group"]
    feat_cols = [c for c in merged.columns if c not in drop_cols
                 and c not in ["readmission_30d", "adverse_event_90d", "mortality_1yr"]
                 and merged[c].dtype in [np.float64, np.int64, float, int]]
    X_struct = merged[feat_cols].fillna(0).values.astype(np.float32)

    # Graph embeddings per patient
    X_graph = np.stack([
        graph_embeddings_map.get(pid, np.zeros(graph_emb_dim))
        for pid in patient_ids
    ]).astype(np.float32)

    # Context embeddings per patient (top_k similar patients)
    X_context = np.stack([
        context_embeddings_map.get(pid, np.zeros((top_k, latent_dim)))
        for pid in patient_ids
    ]).astype(np.float32)

    # Labels
    label_cols = ["readmission_30d", "adverse_event_90d", "mortality_1yr"]
    y = merged[label_cols].values.astype(np.float32)

    logger.info(
        f"Training data shapes — X_struct: {X_struct.shape}, "
        f"X_graph: {X_graph.shape}, X_ctx: {X_context.shape}, y: {y.shape}"
    )
    return X_struct, X_graph, X_context, y, patient_ids
