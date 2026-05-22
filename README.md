# Cortex-RAG 
## Multimodal Knowledge Graph Integration for Longitudinal Clinical Decision Support

Cortex-RAG is a research-grade, end-to-end Python framework that combines a **Clinical Knowledge Graph** with **Retrieval-Augmented Generation (RAG)** to produce explainable, risk-stratified clinical decision support from longitudinal patient data.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CORTEX-RAG PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Multimodal Inputs                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │   Labs   │ │  Notes   │ │ Imaging  │ │ Demographics │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘   │
│       │             │             │               │           │
│       └─────────────┴─────────────┴───────────────┘          │
│                             │                                 │
│                    ┌────────▼────────┐                        │
│                    │ Knowledge Graph │  ◄── Ontology          │
│                    │  (NetworkX)     │                        │
│                    └────────┬────────┘                        │
│                             │                                 │
│              ┌──────────────┼──────────────┐                 │
│              │              │              │                  │
│     ┌────────▼───┐  ┌───────▼──┐  ┌───────▼──────┐          │
│     │  Subgraph  │  │ Similar  │  │  Concept     │          │
│     │  Retrieval │  │ Patients │  │  Embeddings  │          │
│     └────────┬───┘  └───────┬──┘  └───────┬──────┘          │
│              └──────────────┼──────────────┘                 │
│                             │                                 │
│                    ┌────────▼────────┐                        │
│                    │ ClinicalEncoder │  Structured + Graph    │
│                    └────────┬────────┘                        │
│                             │                                 │
│                    ┌────────▼────────┐                        │
│                    │ Cross-Attention │  Patient ↔ Context     │
│                    └────────┬────────┘                        │
│                             │                                 │
│                    ┌────────▼────────┐                        │
│                    │ Prediction Head │  Multi-task outcomes   │
│                    └────────┬────────┘                        │
│                             │                                 │
│         ┌───────────────────┼───────────────────┐            │
│         │                   │                   │            │
│  30-day Readmission  90-day Adverse     1-yr Mortality       │
│  + Risk Score        + Recommendations + Explainability       │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
cortex_rag/
├── main.py               # Orchestration script
├── data_processing.py    # Data simulation, preprocessing, feature engineering
├── knowledge_graph.py    # KG construction, entity extraction, retrieval
├── rag_model.py          # ClinicalEncoder, RAG model, inference system
├── training.py           # Dataset, loss function, training loop, cross-validation
├── evaluation.py         # Metrics, calibration, explainability
├── tests.py              # Unit tests
└── requirements.txt      # Python dependencies
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
cd cortex_rag
python main.py
```

This will:
- Simulate 200 patients with longitudinal lab results, clinical notes, and imaging
- Build a clinical knowledge graph (~4 000 nodes, ~8 000 edges)
- Train the RAG model for 40 epochs
- Print an evaluation report with AUROC / AUPRC / F1 per task
- Run a clinical decision support demo for patient P0001

### 3. Custom Options

```bash
# Larger cohort, more epochs, GPU
python main.py --patients 500 --epochs 80 --device cuda

# Run 3-fold cross-validation
python main.py --cv

# Demo a specific patient
python main.py --demo-patient P0042

# Skip the demo
python main.py --no-demo
```

### 4. Run Unit Tests

```bash
python -m pytest tests.py -v
# or
python tests.py
```

---

## Module Details

### `data_processing.py`
| Function / Class | Description |
|---|---|
| `simulate_patient_demographics()` | Generates realistic patient cohort |
| `simulate_lab_results()` | Longitudinal lab records (5 visits × N patients) |
| `simulate_clinical_notes()` | Free-text notes with template-based NLP features |
| `simulate_imaging_features()` | Mock 128-dim DICOM embeddings + impression text |
| `ClinicalDataPreprocessor` | KNN imputation, Standard scaling, feature engineering |
| `load_and_preprocess()` | Master pipeline returning all processed modalities |

### `knowledge_graph.py`
| Class / Method | Description |
|---|---|
| `ClinicalKnowledgeGraph` | NetworkX DiGraph with typed nodes/edges |
| `build_from_dataframes()` | Populates graph from all data modalities |
| `extract_entities()` | Rule-based NER (diagnoses, medications, symptoms, labs) |
| `get_patient_subgraph()` | k-hop ego-graph retrieval |
| `retrieve_similar_patients()` | Jaccard-based cohort retrieval |
| `compute_node_embeddings()` | Spectral/power-iteration node embeddings |

### `rag_model.py`
| Class | Description |
|---|---|
| `ClinicalRetriever` | Multi-strategy KG-based context retrieval |
| `ClinicalEncoder` | Dual-branch (structured + graph) encoder |
| `MultiheadContextAggregator` | Cross-attention over retrieved patient contexts |
| `ClinicalRAGModel` | Full end-to-end RAG model |
| `CortexRAGSystem` | Inference wrapper + recommendation generator |

### `training.py`
| Class / Function | Description |
|---|---|
| `ClinicalDataset` | PyTorch Dataset for patient records |
| `MultiTaskClinicalLoss` | Weighted BCE + risk regression |
| `RAGTrainer` | Training loop, AdamW, cosine LR, early stopping |
| `run_cross_validation()` | Stratified k-fold CV |
| `prepare_training_data()` | Aligns all modalities into training arrays |

### `evaluation.py`
| Function / Class | Description |
|---|---|
| `compute_metrics()` | Per-task AUROC, AUPRC, F1, Precision, Recall, Brier |
| `expected_calibration_error()` | ECE for calibration analysis |
| `AttentionExplainer` | Attention-weight-based prediction explanations |
| `GradientFeatureImportance` | Integrated gradients for feature attribution |
| `generate_evaluation_report()` | Full plain-text evaluation summary |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **AUROC** | Area under ROC curve (primary discrimination metric) |
| **AUPRC** | Area under Precision-Recall curve (imbalanced classes) |
| **F1** | Harmonic mean of precision and recall |
| **Brier Score** | Probabilistic calibration quality |
| **ECE** | Expected Calibration Error |
| **Retrieval Faithfulness** | Overlap between retrieved context and ground truth |

---

## Ethical & Privacy Considerations

> This project uses **fully synthetic, simulated data**. No real patient data is used or required.

For production deployment, the following safeguards are essential:

- **Data De-identification**: All PHI must be removed or anonymised before processing (HIPAA / GDPR compliant pipelines required).
- **Audit Trails**: Log all model predictions and retrieved contexts for clinical audit.
- **Human-in-the-Loop**: Model outputs are **decision support only** — final clinical decisions must be made by licensed clinicians.
- **Bias Monitoring**: Regularly audit model performance across demographic subgroups.
- **Model Governance**: Maintain model cards, version control, and re-validation schedules.
- **Consent**: Patients should be informed when AI tools contribute to their care decisions.

---

## Extending to Production

| Component | Production Replacement |
|---|---|
| Synthetic data | Real EHR data (Epic, Cerner) via FHIR API |
| Rule-based NER | medspaCy, cTAKES, or fine-tuned BioBERT |
| Graph embeddings | PyG GraphSAGE / GAT |
| Generative component | BioGPT, Med-PaLM, or fine-tuned Llama-3 |
| Vector retrieval | FAISS / Qdrant for embedding-based retrieval |
| Deployment | FastAPI + Docker + Kubernetes |

---

## License

This project is provided for **research and educational purposes**. It is not approved for clinical use without appropriate validation, regulatory review, and institutional approval.

---

*Built with Python · PyTorch · NetworkX · Scikit-learn*
