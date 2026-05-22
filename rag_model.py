"""
rag_model.py
============
Cortex-RAG: Retrieval-Augmented Generation Model

Components:
  1. ClinicalRetriever  – queries the knowledge graph and retrieves relevant context
  2. ClinicalEncoder    – encodes patient state into dense vectors (PyTorch)
  3. ClinicalRAGModel   – end-to-end model fusing retrieved context with patient features
                          for clinical decision support prediction
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Clinical Retriever
# ---------------------------------------------------------------------------

class ClinicalRetriever:
    """
    Retrieves relevant context from the Clinical Knowledge Graph
    given a patient query.
    """

    def __init__(self, knowledge_graph, node_embeddings: Optional[np.ndarray] = None,
                 node_emb_map: Optional[Dict] = None):
        self.kg = knowledge_graph
        self.node_embeddings = node_embeddings   # shape: (n_nodes, dim)
        self.node_emb_map = node_emb_map or {}   # node_id -> embedding vector

    def retrieve_by_patient(
        self, patient_id: str, top_k_similar: int = 3, hops: int = 2
    ) -> Dict:
        """
        Multi-strategy retrieval:
          (a) Subgraph expansion (k-hop neighborhood)
          (b) Similar patient retrieval (Jaccard)
          (c) Concept-level ontological context
        """
        context = {}

        # (a) Ego subgraph context
        subgraph = self.kg.get_patient_subgraph(patient_id, hops=hops)
        context["subgraph_text"] = self.kg.get_node_context(patient_id)
        context["subgraph_size"] = subgraph.number_of_nodes()

        # (b) Similar patients
        similar = self.kg.retrieve_similar_patients(patient_id, top_k=top_k_similar)
        similar_contexts = []
        for sim_pid, score in similar:
            sim_ctx = self.kg.get_node_context(sim_pid)
            similar_contexts.append({"patient_id": sim_pid, "similarity": score,
                                      "context": sim_ctx})
        context["similar_patients"] = similar_contexts

        # (c) Concept embeddings for nodes reachable from patient
        if self.node_emb_map:
            neighbor_ids = list(
                self.kg.get_patient_subgraph(patient_id, hops=1).nodes()
            )
            emb_list = [self.node_emb_map[nid] for nid in neighbor_ids
                        if nid in self.node_emb_map]
            if emb_list:
                context["neighborhood_embedding"] = np.mean(emb_list, axis=0)
            else:
                emb_dim = len(next(iter(self.node_emb_map.values())))
                context["neighborhood_embedding"] = np.zeros(emb_dim)
        else:
            context["neighborhood_embedding"] = np.zeros(64)

        return context

    def retrieve_evidence_text(self, patient_id: str) -> str:
        """
        Build a structured textual evidence summary for the patient
        (used as input to the generative component).
        """
        ctx = self.retrieve_by_patient(patient_id)
        lines = ["=== RETRIEVED CLINICAL EVIDENCE ==="]
        lines.append(f"[Patient Subgraph - {ctx['subgraph_size']} nodes]")
        lines.append(ctx.get("subgraph_text", ""))
        lines.append("\n[Similar Patient Profiles]")
        for sp in ctx.get("similar_patients", []):
            lines.append(
                f"  Sim={sp['similarity']:.3f} | {sp['patient_id']}: "
                + sp["context"].split("\n")[0]
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Clinical Encoder (PyTorch)
# ---------------------------------------------------------------------------

class ClinicalEncoder(nn.Module):
    """
    Multimodal encoder that projects structured clinical features
    and graph neighborhood embeddings into a shared latent space.
    """

    def __init__(
        self,
        structured_dim: int,
        graph_emb_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.structured_encoder = nn.Sequential(
            nn.Linear(structured_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_emb_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(
        self,
        structured_features: torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            structured_features: (batch, structured_dim)
            graph_embedding:     (batch, graph_emb_dim)
        Returns:
            fused_rep: (batch, latent_dim)
        """
        s_enc = self.structured_encoder(structured_features)
        g_enc = self.graph_encoder(graph_embedding)
        fused = self.fusion(torch.cat([s_enc, g_enc], dim=-1))
        return fused


# ---------------------------------------------------------------------------
# 3. Attention-based Context Aggregator
# ---------------------------------------------------------------------------

class MultiheadContextAggregator(nn.Module):
    """
    Cross-attention module that attends over retrieved context embeddings
    weighted by their relevance to the current patient state.
    """

    def __init__(self, latent_dim: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(
        self, query: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query:   (batch, 1, latent_dim)   – current patient representation
            context: (batch, k, latent_dim)   – k retrieved context embeddings
        Returns:
            attended: (batch, latent_dim), attn_weights: (batch, 1, k)
        """
        attended, attn_weights = self.attn(query, context, context)
        attended = self.norm(attended + query)
        attended = attended + self.ff(attended)
        return attended.squeeze(1), attn_weights


# ---------------------------------------------------------------------------
# 4. Clinical RAG Model (full end-to-end)
# ---------------------------------------------------------------------------

class ClinicalRAGModel(nn.Module):
    """
    Full Cortex-RAG model.

    Architecture:
      1. ClinicalEncoder   → encodes (structured features + graph embedding)
      2. Retrieval         → fetches k context embeddings from similar patients
      3. CrossAttention    → attends over retrieved context
      4. Prediction Head   → multi-task head for clinical outcomes
    """

    def __init__(
        self,
        structured_dim: int,
        graph_emb_dim: int = 64,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_tasks: int = 3,      # e.g. readmission, adverse event, mortality
        top_k_retrieve: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.top_k = top_k_retrieve
        self.latent_dim = latent_dim

        self.encoder = ClinicalEncoder(
            structured_dim=structured_dim,
            graph_emb_dim=graph_emb_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.context_aggregator = MultiheadContextAggregator(
            latent_dim=latent_dim, n_heads=n_heads, dropout=dropout
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tasks),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),  # continuous risk score
        )

    def forward(
        self,
        structured_features: torch.Tensor,
        graph_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            structured_features: (batch, structured_dim)
            graph_embeddings:    (batch, graph_emb_dim)
            context_embeddings:  (batch, top_k, latent_dim) – retrieved contexts
        Returns:
            dict with keys: logits, risk_score, attention_weights
        """
        # Encode current patient
        patient_rep = self.encoder(structured_features, graph_embeddings)  # (B, L)

        # Cross-attend over retrieved context
        query = patient_rep.unsqueeze(1)  # (B, 1, L)
        attended, attn_weights = self.context_aggregator(query, context_embeddings)

        # Concatenate patient rep with attended context
        combined = torch.cat([patient_rep, attended], dim=-1)  # (B, 2L)

        # Predict
        logits = self.prediction_head(combined)        # (B, n_tasks)
        risk_score = self.risk_head(combined).squeeze(-1)  # (B,)

        return {
            "logits": logits,
            "risk_score": risk_score,
            "attention_weights": attn_weights,
            "patient_embedding": patient_rep,
        }


# ---------------------------------------------------------------------------
# 5. RAG System Wrapper (ties retriever + model together)
# ---------------------------------------------------------------------------

class CortexRAGSystem:
    """
    End-to-end Cortex-RAG inference system.
    Combines retrieval from the knowledge graph with the neural RAG model.
    """

    def __init__(
        self,
        model: ClinicalRAGModel,
        retriever: ClinicalRetriever,
        patient_embeddings: np.ndarray,        # shape: (n_patients, latent_dim)
        patient_id_index: Dict[str, int],      # patient_id -> row index in embeddings
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.retriever = retriever
        self.patient_embeddings = patient_embeddings
        self.pid_index = patient_id_index
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        patient_id: str,
        structured_features: np.ndarray,
        graph_embedding: np.ndarray,
    ) -> Dict:
        """
        Full inference pipeline for a single patient.
        Returns predictions + explainability outputs.
        """
        # Retrieve similar-patient context
        ctx = self.retriever.retrieve_by_patient(patient_id, top_k_similar=self.model.top_k)
        similar_pids = [sp["patient_id"] for sp in ctx["similar_patients"]]

        # Build context tensor from similar-patient embeddings
        context_embs = []
        for spid in similar_pids:
            if spid in self.pid_index:
                context_embs.append(self.patient_embeddings[self.pid_index[spid]])
        while len(context_embs) < self.model.top_k:
            context_embs.append(np.zeros(self.model.latent_dim))
        context_tensor = torch.tensor(
            np.stack(context_embs[:self.model.top_k]), dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # (1, top_k, latent_dim)

        # Model forward
        sf = torch.tensor(structured_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        ge = torch.tensor(graph_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)

        outputs = self.model(sf, ge, context_tensor)

        probs = torch.sigmoid(outputs["logits"]).cpu().numpy()[0]
        risk = torch.sigmoid(outputs["risk_score"]).item()
        attn = outputs["attention_weights"].cpu().numpy()[0, 0]

        return {
            "patient_id": patient_id,
            "risk_score": round(risk, 4),
            "readmission_30d_prob": round(float(probs[0]), 4),
            "adverse_event_90d_prob": round(float(probs[1]), 4),
            "mortality_1yr_prob": round(float(probs[2]), 4),
            "retrieved_similar_patients": [
                {"patient_id": spid, "attention_weight": float(attn[i])}
                for i, spid in enumerate(similar_pids[: self.model.top_k])
            ],
            "evidence_text": self.retriever.retrieve_evidence_text(patient_id),
        }

    def generate_recommendation(self, prediction: Dict) -> str:
        """
        Rule-based clinical recommendation generator based on risk scores.
        In production, replace with a fine-tuned clinical LLM (e.g., Med-PaLM, BioGPT).
        """
        lines = [f"=== CORTEX-RAG CLINICAL DECISION SUPPORT ===",
                 f"Patient: {prediction['patient_id']}",
                 f"Composite Risk Score: {prediction['risk_score']:.2%}",
                 ""]

        risk = prediction["risk_score"]
        r30 = prediction["readmission_30d_prob"]
        ae90 = prediction["adverse_event_90d_prob"]
        mort = prediction["mortality_1yr_prob"]

        lines.append("Risk Assessment:")
        lines.append(f"  • 30-day Readmission:    {r30:.1%}")
        lines.append(f"  • 90-day Adverse Event:  {ae90:.1%}")
        lines.append(f"  • 1-year Mortality:       {mort:.1%}")
        lines.append("")

        # Tiered recommendations
        lines.append("Clinical Recommendations:")
        if risk > 0.6:
            lines += [
                "  [HIGH RISK] Immediate clinical attention warranted.",
                "  • Schedule care management enrollment within 48 hours.",
                "  • Arrange multidisciplinary team review.",
                "  • Consider hospitalist or specialist referral.",
                "  • Escalate monitoring frequency to weekly.",
            ]
        elif risk > 0.35:
            lines += [
                "  [MODERATE RISK] Proactive monitoring recommended.",
                "  • Schedule follow-up within 2 weeks.",
                "  • Review and optimise polypharmacy.",
                "  • Refer to disease management programme.",
                "  • Repeat key labs in 4-6 weeks.",
            ]
        else:
            lines += [
                "  [LOW RISK] Routine follow-up appropriate.",
                "  • Continue current care plan.",
                "  • Routine follow-up in 3 months.",
                "  • Reinforce lifestyle counseling.",
            ]

        lines.append("")
        lines.append("Evidence Basis (Retrieved Context):")
        lines.append(
            "\n".join("  " + l for l in prediction["evidence_text"].split("\n")[:10])
        )

        lines.append("\n[DISCLAIMER] AI-generated support only. "
                     "Clinical judgment supersedes this output.")
        return "\n".join(lines)
