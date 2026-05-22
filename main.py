"""
main.py
=======
Cortex-RAG: Main Orchestration Script

Runs the full end-to-end pipeline:
  1. Data acquisition and preprocessing
  2. Knowledge graph construction + embeddings
  3. Model initialisation
  4. Training with cross-validation
  5. Evaluation and report generation
  6. Clinical decision support demo

Usage:
    python main.py [--patients N] [--epochs E] [--device cpu|cuda] [--demo-patient P0001]
"""

import argparse
import logging
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────────────────────
from data_processing import load_and_preprocess
from evaluation import (
    AttentionExplainer,
    generate_evaluation_report,
    save_evaluation_plots,
    tune_task_thresholds,
)
from knowledge_graph import ClinicalKnowledgeGraph
from rag_model import ClinicalRAGModel, ClinicalRetriever, CortexRAGSystem
from training import (
    ClinicalDataset,
    RAGTrainer,
    prepare_training_data,
    run_cross_validation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ─────────────────────────────────────────────────────────────────────────────
# Config defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "n_patients": 500,  # Increased from 200 for more training signal
    "n_epochs": 50,     # Increased from 40
    "batch_size": 32,
    "lr": 2.5e-4,       # Fine-tuned learning rate
    "latent_dim": 256,  # Doubled from 128 for more capacity
    "graph_emb_dim": 64,
    "hidden_dim": 512,  # Doubled from 256 for deeper encoders
    "n_heads": 8,       # Doubled from 4 for more attention heads
    "top_k": 3,
    "dropout": 0.35,    # Increased from 0.2 for better regularization
    "n_cv_folds": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "demo_patient": "P0001",
    "run_cv": False,  # set True for full cross-validation (slower)
    "min_precision": 0.40,
    "eval_output_dir": "outputs/evaluation",
    "early_stop_patience": 20,
    "weight_decay": 1e-3,      # L2 regularization
    "l1_penalty": 1e-5,        # L1 regularization
    "mixup_alpha": 0.2,        # Mixup augmentation strength
    "task_weights": [1.0, 1.8, 2.5],  # Optimized task importance weights
    "ensemble_size": 3,        # Number of models to ensemble
}


# ─────────────────────────────────────────────────────────────────────────────
# Step helpers
# ─────────────────────────────────────────────────────────────────────────────

def step_data(cfg: Dict):
    logger.info("STEP 1: Data Acquisition & Preprocessing")
    data = load_and_preprocess(n_patients=cfg["n_patients"])
    logger.info(f"  Patients      : {len(data['demographics'])}")
    logger.info(f"  Lab records   : {len(data['labs'])}")
    logger.info(f"  Clinical notes: {len(data['notes'])}")
    logger.info(f"  Imaging rows  : {len(data['imaging'])}")
    logger.info(f"  Feature matrix: {data['patient_features'].shape}")
    return data


def step_knowledge_graph(data: Dict):
    logger.info("STEP 2: Knowledge Graph Construction")
    kg = ClinicalKnowledgeGraph()
    kg.build_from_dataframes(
        data["demographics"],
        data["labs"],
        data["notes"],
        data["imaging"],
    )
    stats = kg.get_stats()
    logger.info(f"  Nodes: {stats['total_nodes']}  Edges: {stats['total_edges']}")
    logger.info(f"  Node types : {stats['node_types']}")
    logger.info(f"  Edge types : {stats['edge_types']}")

    logger.info("  Computing node embeddings ...")
    emb_matrix, emb_map = kg.compute_node_embeddings(dim=64)
    return kg, emb_matrix, emb_map


def step_prepare_data(data: Dict, kg, emb_map: Dict, cfg: Dict):
    """Build training tensors from the knowledge graph + patient features."""
    logger.info("╔══ STEP 3: Preparing Training Data ══╗")

    # Graph embedding per patient: mean of their k-hop neighbourhood embeddings
    patient_ids = data["demographics"]["patient_id"].tolist()
    graph_embeddings_map: Dict = {}
    for pid in patient_ids:
        subgraph = kg.get_patient_subgraph(pid, hops=1)
        neighbor_embs = [emb_map[n] for n in subgraph.nodes() if n in emb_map]
        graph_embeddings_map[pid] = (
            np.mean(neighbor_embs, axis=0) if neighbor_embs else np.zeros(cfg["graph_emb_dim"])
        )

    # Retrieve similar patient context embeddings
    context_embeddings_map: Dict = {}
    for pid in patient_ids:
        similar = kg.retrieve_similar_patients(pid, top_k=cfg["top_k"])
        ctx_list = []
        for spid, _ in similar:
            ctx_list.append(graph_embeddings_map.get(spid, np.zeros(cfg["graph_emb_dim"])))
        while len(ctx_list) < cfg["top_k"]:
            ctx_list.append(np.zeros(cfg["graph_emb_dim"]))
        # Project to latent_dim with zeros padding
        padded = np.stack(ctx_list[: cfg["top_k"]])
        padding = np.zeros((cfg["top_k"], cfg["latent_dim"] - cfg["graph_emb_dim"]))
        context_embeddings_map[pid] = np.concatenate([padded, padding], axis=1)

    X_struct, X_graph, X_ctx, y, aligned_ids = prepare_training_data(
        patient_features=data["patient_features"],
        outcomes=data["outcomes"],
        graph_embeddings_map=graph_embeddings_map,
        context_embeddings_map=context_embeddings_map,
        top_k=cfg["top_k"],
        latent_dim=cfg["latent_dim"],
        graph_emb_dim=cfg["graph_emb_dim"],
    )
    return X_struct, X_graph, X_ctx, y, aligned_ids, graph_embeddings_map, context_embeddings_map


def step_train(X_struct, X_graph, X_ctx, y, cfg: Dict):
    logger.info("╔══ STEP 4: Model Training ══╗")

    # Train / validation split (stratify on first task)
    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y[:, 0].astype(int),
                                       random_state=42)

    # Fit structured-feature scaling on train only to avoid validation leakage.
    from sklearn.preprocessing import StandardScaler

    struct_scaler = StandardScaler()
    X_struct = X_struct.copy()
    X_struct[tr_idx] = struct_scaler.fit_transform(X_struct[tr_idx])
    X_struct[va_idx] = struct_scaler.transform(X_struct[va_idx])

    structured_dim = X_struct.shape[1]
    logger.info(f"  structured_dim={structured_dim}  device={cfg['device']}")

    def model_factory():
        return ClinicalRAGModel(
            structured_dim=structured_dim,
            graph_emb_dim=cfg["graph_emb_dim"],
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_heads=cfg["n_heads"],
            n_tasks=3,
            top_k_retrieve=cfg["top_k"],
            dropout=cfg["dropout"],
        )

    # ── Optional cross-validation ─────────────────────────────────────────
    cv_results = None
    if cfg.get("run_cv", False):
        logger.info("  Running cross-validation ...")
        cv_results = run_cross_validation(
            model_factory=model_factory,
            patient_features=X_struct,
            graph_embeddings=X_graph,
            context_embeddings=X_ctx,
            labels=y,
            n_splits=cfg["n_cv_folds"],
            n_epochs=cfg["n_epochs"] // 2,
            batch_size=cfg["batch_size"],
            device=cfg["device"],
        )
        logger.info(f"  CV Aggregated: {cv_results['aggregated']}")

    # ── Final model on full train set ────────────────────────────────────
    train_ds = ClinicalDataset(X_struct[tr_idx], X_graph[tr_idx], X_ctx[tr_idx], y[tr_idx])
    val_ds = ClinicalDataset(X_struct[va_idx], X_graph[va_idx], X_ctx[va_idx], y[va_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = model_factory()
    trainer = RAGTrainer(
        model,
        train_labels=y[tr_idx],
        device=cfg["device"],
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-3),
        l1_penalty=cfg.get("l1_penalty", 1e-5),
        patience=cfg.get("early_stop_patience", 20),
        task_weights=cfg.get("task_weights", [1.0, 1.5, 2.0]),
        mixup_alpha=cfg.get("mixup_alpha", 0.2),
    )
    history = trainer.train(train_loader, val_loader, n_epochs=cfg["n_epochs"])

    logger.info(
        f"  Training complete. Best val AUC: "
        f"{max(history['val_auc']):.4f}"
    )
    
    # Train ensemble if configured
    models = [model]
    if cfg.get("ensemble_size", 1) > 1:
        logger.info(f"  Training ensemble of {cfg['ensemble_size']} models...")
        for i in range(1, cfg["ensemble_size"]):
            logger.info(f"    Ensemble model {i+1}/{cfg['ensemble_size']}...")
            ens_model = model_factory()
            ens_trainer = RAGTrainer(
                ens_model,
                train_labels=y[tr_idx],
                device=cfg["device"],
                lr=cfg["lr"],
                weight_decay=cfg.get("weight_decay", 1e-3),
                l1_penalty=cfg.get("l1_penalty", 1e-5),
                patience=cfg.get("early_stop_patience", 20),
                task_weights=cfg.get("task_weights", [1.0, 1.5, 2.0]),
                mixup_alpha=cfg.get("mixup_alpha", 0.2),
            )
            ens_trainer.train(train_loader, val_loader, n_epochs=cfg["n_epochs"])
            models.append(ens_model)
    
    return models, trainer, va_idx, cv_results


def step_evaluate(models, X_struct, X_graph, X_ctx, y, va_idx, aligned_ids, cfg: Dict):
    logger.info("╔══ STEP 5: Evaluation ══╗")
    # Handle single model or ensemble
    if not isinstance(models, list):
        models = [models]
    
    all_labels = []
    ensemble_probs = None  # Will accumulate predictions

    val_ds = ClinicalDataset(X_struct[va_idx], X_graph[va_idx], X_ctx[va_idx], y[va_idx])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    with torch.no_grad():
        for model in models:
            model.eval()
            model_probs = []
            for sf, ge, ctx_emb, labels in val_loader:
                sf = sf.to(cfg["device"])
                ge = ge.to(cfg["device"])
                ctx_emb = ctx_emb.to(cfg["device"])
                out = model(sf, ge, ctx_emb)
                probs = torch.sigmoid(out["logits"]).cpu().numpy()
                model_probs.append(probs)
                if ensemble_probs is None:
                    all_labels.append(labels.numpy())
            
            model_probs = np.vstack(model_probs)
            if ensemble_probs is None:
                ensemble_probs = model_probs / len(models)
            else:
                ensemble_probs += model_probs / len(models)

    all_probs = ensemble_probs
    all_labels = np.vstack(all_labels)

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    val_ids = [aligned_ids[i] for i in va_idx]

    thresholds = tune_task_thresholds(
        all_labels,
        all_probs,
        min_precision=cfg.get("min_precision", 0.25),
    )
    logger.info(f"  Tuned thresholds (best F1 subject to precision floor): {thresholds}")

    report = generate_evaluation_report(
        all_labels,
        all_probs,
        val_ids,
        thresholds=thresholds,
    )
    plot_files = save_evaluation_plots(
        all_labels,
        all_probs,
        thresholds=thresholds,
        output_dir=cfg.get("eval_output_dir", "outputs/evaluation"),
    )
    print(report)
    if plot_files:
        logger.info("  Saved evaluation plots:")
        for path in plot_files:
            logger.info(f"    - {path}")
    return report


def step_demo(
    model, kg: ClinicalKnowledgeGraph, emb_map: Dict,
    graph_embeddings_map: Dict, context_embeddings_map: Dict,
    aligned_ids, X_struct, cfg: Dict,
):
    logger.info("╔══ STEP 6: Clinical Decision Support Demo ══╗")

    demo_pid = cfg["demo_patient"]
    if demo_pid not in graph_embeddings_map:
        demo_pid = aligned_ids[0]
        logger.warning(f"  Demo patient {cfg['demo_patient']} not found; using {demo_pid}.")

    idx = aligned_ids.index(demo_pid)
    sf_np = X_struct[idx]
    ge_np = graph_embeddings_map[demo_pid]
    ctx_np = context_embeddings_map[demo_pid]

    # Build patient embedding store from trained model encoder
    model.eval()
    with torch.no_grad():
        all_sf = torch.tensor(X_struct, dtype=torch.float32).to(cfg["device"])
        all_ge = torch.tensor(
            np.stack([graph_embeddings_map.get(pid, np.zeros(cfg["graph_emb_dim"]))
                      for pid in aligned_ids]),
            dtype=torch.float32
        ).to(cfg["device"])
        patient_embs = model.encoder(all_sf, all_ge).cpu().numpy()

    pid_to_idx = {pid: i for i, pid in enumerate(aligned_ids)}
    retriever = ClinicalRetriever(kg, node_emb_map=emb_map)

    rag_system = CortexRAGSystem(
        model=model,
        retriever=retriever,
        patient_embeddings=patient_embs,
        patient_id_index=pid_to_idx,
        device=cfg["device"],
    )

    prediction = rag_system.predict(demo_pid, sf_np, ge_np)
    recommendation = rag_system.generate_recommendation(prediction)

    print("\n" + "=" * 70)
    print(recommendation)
    print("=" * 70)

    # Explainability
    explainer = AttentionExplainer()
    explanation = explainer.explain_prediction(prediction)
    print("\n" + explanation["explanation_text"])

    return prediction, recommendation


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Cortex-RAG Clinical Decision Support")
    p.add_argument("--patients", type=int, default=DEFAULT_CONFIG["n_patients"])
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["n_epochs"])
    p.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    p.add_argument("--demo-patient", type=str, default=DEFAULT_CONFIG["demo_patient"])
    p.add_argument("--cv", action="store_true", help="Run cross-validation")
    p.add_argument("--no-demo", action="store_true", help="Skip clinical demo")
    p.add_argument(
        "--min-precision",
        type=float,
        default=DEFAULT_CONFIG["min_precision"],
        help="Minimum precision target used for threshold tuning",
    )
    p.add_argument(
        "--eval-output-dir",
        type=str,
        default=DEFAULT_CONFIG["eval_output_dir"],
        help="Directory for ROC/PR/confusion matrix plots",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {**DEFAULT_CONFIG}
    cfg["n_patients"] = args.patients
    cfg["n_epochs"] = args.epochs
    cfg["device"] = args.device
    cfg["demo_patient"] = args.demo_patient
    cfg["run_cv"] = args.cv
    cfg["min_precision"] = args.min_precision
    cfg["eval_output_dir"] = args.eval_output_dir

    t0 = time.time()
    logger.info(" Cortex-RAG: Multimodal Knowledge Graph RAG for Clinical Decision Support")
    logger.info(f"    Patients={cfg['n_patients']}  Epochs={cfg['n_epochs']}  Device={cfg['device']}")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    data = step_data(cfg)
    kg, emb_matrix, emb_map = step_knowledge_graph(data)
    X_struct, X_graph, X_ctx, y, aligned_ids, ge_map, ctx_map = step_prepare_data(
        data, kg, emb_map, cfg
    )
    models, trainer, va_idx, cv_results = step_train(X_struct, X_graph, X_ctx, y, cfg)
    step_evaluate(models, X_struct, X_graph, X_ctx, y, va_idx, aligned_ids, cfg)

    if not args.no_demo:
        # Use first model from ensemble for demo
        primary_model = models[0] if isinstance(models, list) else models
        step_demo(primary_model, kg, emb_map, ge_map, ctx_map, aligned_ids, X_struct, cfg)

    elapsed = time.time() - t0
    logger.info(f" Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
