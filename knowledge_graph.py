"""
knowledge_graph.py
==================
Cortex-RAG: Multimodal Knowledge Graph Construction Module

Builds a clinical knowledge graph integrating:
  - Patient nodes (demographics, longitudinal lab trends)
  - Concept nodes (diagnoses, medications, symptoms, lab tests)
  - Relationship edges (HAS_CONDITION, PRESCRIBED, EXHIBITS, MEASURED_BY, etc.)
  - Graph Neural Network (GNN) embeddings for downstream retrieval
"""

import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ontology / Schema Definitions
# ---------------------------------------------------------------------------

NODE_TYPES = {
    "PATIENT": "patient",
    "DIAGNOSIS": "diagnosis",
    "MEDICATION": "medication",
    "SYMPTOM": "symptom",
    "LAB_TEST": "lab_test",
    "IMAGING": "imaging",
    "PROVIDER": "provider",
}

EDGE_TYPES = {
    "HAS_CONDITION": "has_condition",
    "PRESCRIBED": "prescribed",
    "EXHIBITS": "exhibits",
    "MEASURED_BY": "measured_by",
    "HAS_IMAGING": "has_imaging",
    "TREATED_BY": "treated_by",
    "RELATED_TO": "related_to",       # concept–concept links
    "COMORBID_WITH": "comorbid_with",  # patient-level comorbidity co-occurrence
}

# Minimal clinical ontology for concept-concept edges
CLINICAL_ONTOLOGY = [
    ("Type 2 Diabetes", "Hypertension", "COMORBID_WITH"),
    ("Type 2 Diabetes", "Chronic Kidney Disease", "COMORBID_WITH"),
    ("Hypertension", "Coronary Artery Disease", "COMORBID_WITH"),
    ("Heart Failure", "Atrial Fibrillation", "COMORBID_WITH"),
    ("Metformin", "Type 2 Diabetes", "TREATS"),
    ("Lisinopril", "Hypertension", "TREATS"),
    ("Lisinopril", "Heart Failure", "TREATS"),
    ("Atorvastatin", "Coronary Artery Disease", "TREATS"),
    ("Furosemide", "Heart Failure", "TREATS"),
    ("Warfarin", "Atrial Fibrillation", "TREATS"),
    ("HbA1c", "Type 2 Diabetes", "DIAGNOSTIC_FOR"),
    ("eGFR", "Chronic Kidney Disease", "DIAGNOSTIC_FOR"),
    ("BNP", "Heart Failure", "DIAGNOSTIC_FOR"),
    ("LDL", "Coronary Artery Disease", "DIAGNOSTIC_FOR"),
]

# Rule-based entity extraction patterns
ENTITY_PATTERNS = {
    "diagnosis": re.compile(
        r"(diabetes|hypertension|heart failure|COPD|atrial fibrillation|"
        r"coronary artery disease|chronic kidney disease|hypothyroidism|osteoarthritis)",
        re.IGNORECASE,
    ),
    "medication": re.compile(
        r"(metformin|lisinopril|atorvastatin|aspirin|amlodipine|metoprolol|"
        r"furosemide|levothyroxine|omeprazole|warfarin|insulin)",
        re.IGNORECASE,
    ),
    "symptom": re.compile(
        r"(fatigue|shortness of breath|chest pain|palpitations|leg swelling|"
        r"dizziness|headache|polyuria|polydipsia|dyspnea)",
        re.IGNORECASE,
    ),
    "lab_test": re.compile(
        r"(HbA1c|creatinine|eGFR|BNP|LDL|glucose|cholesterol|WBC|Hgb|TSH|potassium)",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

class ClinicalKnowledgeGraph:
    """
    Constructs and manages a NetworkX-based multimodal clinical knowledge graph.
    Provides retrieval methods for patient-context subgraph extraction.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._node_index: Dict[str, int] = {}
        self._embeddings: Optional[np.ndarray] = None

    # ----- Node / Edge Helpers -----

    def _add_node(self, node_id: str, node_type: str, **attrs):
        if node_id not in self.graph:
            self.graph.add_node(node_id, node_type=node_type, **attrs)
            self._node_index[node_id] = len(self._node_index)

    def _add_edge(self, src: str, dst: str, edge_type: str, **attrs):
        if src in self.graph and dst in self.graph:
            self.graph.add_edge(src, dst, edge_type=edge_type, **attrs)

    # ----- Ontology Loading -----

    def load_ontology(self):
        """Seed graph with concept-concept ontological edges."""
        for subj, obj, rel in CLINICAL_ONTOLOGY:
            # Determine subject type
            subj_type = "medication" if any(
                subj.lower() in m.lower()
                for m in ["metformin", "lisinopril", "atorvastatin", "aspirin",
                           "amlodipine", "metoprolol", "furosemide",
                           "levothyroxine", "omeprazole", "warfarin", "insulin"]
            ) else "diagnosis"

            # Determine object type based on relation type
            obj_type = "lab_test" if rel == "DIAGNOSTIC_FOR" else "diagnosis"

            self._add_node(subj, subj_type, label=subj)
            self._add_node(obj, obj_type, label=obj)
            self._add_edge(subj, obj, rel)

        logger.info(f"Ontology loaded: {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges")

    # ----- Entity Extraction from Notes -----

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Rule-based named entity recognition on clinical text."""
        entities: Dict[str, List[str]] = defaultdict(list)
        for entity_type, pattern in ENTITY_PATTERNS.items():
            matches = pattern.findall(text)
            entities[entity_type].extend([m.lower() for m in matches])
        return dict(entities)

    # ----- Build Patient Subgraph -----

    def add_patient(
        self,
        patient_id: str,
        demographics: Dict,
        lab_record: Optional[Dict] = None,
        note_text: Optional[str] = None,
        imaging_record: Optional[Dict] = None,
    ):
        """Add a patient node and all linked clinical entities."""
        self._add_node(
            patient_id,
            NODE_TYPES["PATIENT"],
            age=demographics.get("age"),
            gender=demographics.get("gender"),
            bmi=demographics.get("bmi"),
            smoker=demographics.get("smoker"),
        )

        # Lab nodes + edges
        if lab_record:
            for lab_name in ["hba1c", "creatinine", "glucose", "cholesterol_ldl",
                             "systolic_bp", "egfr", "bnp"]:
                val = lab_record.get(lab_name)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    lab_node = f"LAB_{lab_name.upper()}"
                    self._add_node(lab_node, NODE_TYPES["LAB_TEST"], label=lab_name)
                    self._add_edge(
                        patient_id, lab_node, EDGE_TYPES["MEASURED_BY"],
                        value=val, date=lab_record.get("visit_date", "")
                    )

        # NLP extraction from notes
        if note_text:
            entities = self.extract_entities(note_text)
            for dx in entities.get("diagnosis", []):
                dx_node = f"DX_{dx.upper().replace(' ', '_')}"
                self._add_node(dx_node, NODE_TYPES["DIAGNOSIS"], label=dx)
                self._add_edge(patient_id, dx_node, EDGE_TYPES["HAS_CONDITION"])

            for med in entities.get("medication", []):
                med_node = f"MED_{med.upper()}"
                self._add_node(med_node, NODE_TYPES["MEDICATION"], label=med)
                self._add_edge(patient_id, med_node, EDGE_TYPES["PRESCRIBED"])

            for sym in entities.get("symptom", []):
                sym_node = f"SYM_{sym.upper().replace(' ', '_')}"
                self._add_node(sym_node, NODE_TYPES["SYMPTOM"], label=sym)
                self._add_edge(patient_id, sym_node, EDGE_TYPES["EXHIBITS"])

        # Imaging node
        if imaging_record:
            img_node = f"IMG_{patient_id}_{imaging_record.get('modality', 'unknown').replace(' ', '_')}"
            self._add_node(
                img_node, NODE_TYPES["IMAGING"],
                modality=imaging_record.get("modality"),
                impression=imaging_record.get("impression"),
                embedding=imaging_record.get("embedding"),
            )
            self._add_edge(patient_id, img_node, EDGE_TYPES["HAS_IMAGING"])

    # ----- Build Full Graph from DataFrames -----

    def build_from_dataframes(
        self,
        demographics: pd.DataFrame,
        labs: pd.DataFrame,
        notes: pd.DataFrame,
        imaging: Optional[pd.DataFrame] = None,
    ):
        """Populate the full knowledge graph from processed DataFrames."""
        self.load_ontology()

        imaging_map = {}
        if imaging is not None and not imaging.empty:
            for _, row in imaging.iterrows():
                imaging_map.setdefault(row["patient_id"], []).append(row.to_dict())

        # Use most-recent lab record per patient
        labs_latest = (
            labs.sort_values("visit_date").groupby("patient_id").last().reset_index()
        )
        labs_map = labs_latest.set_index("patient_id").to_dict("index")

        # Concatenate notes per patient
        notes_map = notes.groupby("patient_id")["note_text"].apply(" ".join).to_dict()

        for _, dem_row in demographics.iterrows():
            pid = dem_row["patient_id"]
            self.add_patient(
                patient_id=pid,
                demographics=dem_row.to_dict(),
                lab_record=labs_map.get(pid),
                note_text=notes_map.get(pid, ""),
                imaging_record=imaging_map.get(pid, [None])[0],
            )

        logger.info(
            f"Graph built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    # ----- Retrieval -----

    def get_patient_subgraph(self, patient_id: str, hops: int = 2) -> nx.DiGraph:
        """Return ego-graph (k-hop neighborhood) for a patient."""
        if patient_id not in self.graph:
            logger.warning(f"Patient {patient_id} not found in graph.")
            return nx.DiGraph()
        nodes = nx.ego_graph(self.graph, patient_id, radius=hops, undirected=True).nodes()
        return self.graph.subgraph(nodes).copy()

    def retrieve_similar_patients(
        self, patient_id: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find patients with overlapping diagnoses/medications in the graph.
        Returns list of (patient_id, jaccard_similarity).
        """
        if patient_id not in self.graph:
            return []
        query_neighbors = set(self.graph.successors(patient_id))
        similar = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") == NODE_TYPES["PATIENT"] and node != patient_id:
                candidate_neighbors = set(self.graph.successors(node))
                union = query_neighbors | candidate_neighbors
                intersection = query_neighbors & candidate_neighbors
                sim = len(intersection) / max(len(union), 1)
                similar.append((node, sim))
        return sorted(similar, key=lambda x: x[1], reverse=True)[:top_k]

    def get_node_context(self, node_id: str) -> str:
        """Serialise a node and its immediate neighbors as a context string."""
        if node_id not in self.graph:
            return ""
        attrs = self.graph.nodes[node_id]
        parts = [f"Node: {node_id} [{attrs.get('node_type', 'unknown')}]"]
        for _, neighbor, edge_data in self.graph.out_edges(node_id, data=True):
            n_attrs = self.graph.nodes[neighbor]
            label = n_attrs.get("label", neighbor)
            etype = edge_data.get("edge_type", "")
            val = edge_data.get("value", "")
            val_str = f" = {val:.2f}" if isinstance(val, float) else ""
            parts.append(f"  --[{etype}]--> {label}{val_str}")
        return "\n".join(parts)

    # ----- Graph Embedding (simple spectral / random walk) -----

    def compute_node_embeddings(self, dim: int = 64) -> np.ndarray:
        """
        Lightweight node2vec-style random walk embedding using power-iteration.
        In production, replace with PyG GraphSAGE or GAT.
        """
        n = self.graph.number_of_nodes()
        if n == 0:
            return np.array([])

        nodes = list(self.graph.nodes())
        idx = {node: i for i, node in enumerate(nodes)}

        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.float32)
        for u, v in self.graph.edges():
            A[idx[u], idx[v]] = 1.0
            A[idx[v], idx[u]] = 1.0  # symmetrise for embedding

        # Degree-normalised Laplacian-based init
        degree = A.sum(axis=1, keepdims=True).clip(min=1)
        A_norm = A / degree

        # Random initialisation + power iteration
        np.random.seed(42)
        E = np.random.randn(n, dim).astype(np.float32)
        for _ in range(10):
            E = A_norm @ E
            # Orthonormalise via QR
            E, _ = np.linalg.qr(E)

        self._embeddings = E
        node_emb_map = {node: E[i] for node, i in idx.items()}
        logger.info(f"Node embeddings computed: shape {E.shape}")
        return E, node_emb_map

    def get_stats(self) -> Dict:
        """Return summary statistics about the graph."""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)
        for _, attrs in self.graph.nodes(data=True):
            node_types[attrs.get("node_type", "unknown")] += 1
        for _, _, attrs in self.graph.edges(data=True):
            edge_types[attrs.get("edge_type", "unknown")] += 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
        }
