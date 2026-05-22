"""
tests.py
========
Cortex-RAG: Unit Tests

Covers:
  - Data simulation and preprocessing
  - Knowledge graph construction and retrieval
  - RAG model forward pass
  - Evaluation metrics
"""

import json
import unittest

import numpy as np
import pandas as pd
import torch


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        from data_processing import (
            simulate_patient_demographics,
            simulate_lab_results,
            simulate_clinical_notes,
            simulate_imaging_features,
            simulate_outcomes,
            ClinicalDataPreprocessor,
        )
        self.n = 20
        self.demographics = simulate_patient_demographics(self.n)
        self.labs = simulate_lab_results(self.demographics, visits_per_patient=3)
        self.notes = simulate_clinical_notes(self.demographics, n_notes_per_patient=2)
        self.imaging = simulate_imaging_features(self.demographics)
        self.outcomes = simulate_outcomes(self.demographics)
        self.preprocessor = ClinicalDataPreprocessor()

    def test_demographics_shape(self):
        self.assertEqual(len(self.demographics), self.n)
        self.assertIn("patient_id", self.demographics.columns)
        self.assertIn("age", self.demographics.columns)

    def test_labs_shape(self):
        self.assertEqual(len(self.labs), self.n * 3)

    def test_preprocess_labs(self):
        df_clean, scaled = self.preprocessor.preprocess_labs(self.labs, fit=True)
        self.assertEqual(len(df_clean), len(self.labs))
        self.assertFalse(np.any(np.isnan(scaled)))

    def test_preprocess_demographics(self):
        df = self.preprocessor.preprocess_demographics(self.demographics)
        self.assertIn("gender_enc", df.columns)
        self.assertIn("obesity", df.columns)

    def test_note_features(self):
        notes = self.preprocessor.extract_note_features(self.notes)
        self.assertIn("mention_diabetes", notes.columns)
        self.assertIn("note_length", notes.columns)

    def test_aggregate_features(self):
        dem = self.preprocessor.preprocess_demographics(self.demographics)
        labs, _ = self.preprocessor.preprocess_labs(self.labs)
        notes = self.preprocessor.extract_note_features(self.notes)
        pf = self.preprocessor.aggregate_patient_features(dem, labs, notes, self.imaging)
        self.assertEqual(len(pf), self.n)
        # No all-NaN columns
        self.assertFalse(pf.isnull().all(axis=0).any())


class TestKnowledgeGraph(unittest.TestCase):

    def setUp(self):
        from data_processing import (
            simulate_patient_demographics,
            simulate_lab_results,
            simulate_clinical_notes,
            simulate_imaging_features,
            ClinicalDataPreprocessor,
        )
        from knowledge_graph import ClinicalKnowledgeGraph

        n = 15
        dem = simulate_patient_demographics(n)
        labs = simulate_lab_results(dem, visits_per_patient=2)
        notes = simulate_clinical_notes(dem, n_notes_per_patient=2)
        imaging = simulate_imaging_features(dem)
        prep = ClinicalDataPreprocessor()
        dem = prep.preprocess_demographics(dem)
        labs, _ = prep.preprocess_labs(labs)
        notes = prep.extract_note_features(notes)

        self.kg = ClinicalKnowledgeGraph()
        self.kg.build_from_dataframes(dem, labs, notes, imaging)
        self.dem = dem

    def test_graph_non_empty(self):
        stats = self.kg.get_stats()
        self.assertGreater(stats["total_nodes"], 0)
        self.assertGreater(stats["total_edges"], 0)

    def test_patient_nodes_exist(self):
        for pid in self.dem["patient_id"]:
            self.assertIn(pid, self.kg.graph.nodes)

    def test_subgraph_retrieval(self):
        pid = self.dem["patient_id"].iloc[0]
        sg = self.kg.get_patient_subgraph(pid, hops=2)
        self.assertGreaterEqual(sg.number_of_nodes(), 1)

    def test_similar_patients(self):
        pid = self.dem["patient_id"].iloc[0]
        similar = self.kg.retrieve_similar_patients(pid, top_k=3)
        self.assertIsInstance(similar, list)

    def test_entity_extraction(self):
        text = "Patient has diabetes and hypertension, prescribed Metformin."
        entities = self.kg.extract_entities(text)
        self.assertIn("diagnosis", entities)
        self.assertIn("medication", entities)

    def test_node_embeddings(self):
        embs, emb_map = self.kg.compute_node_embeddings(dim=16)
        self.assertEqual(embs.shape[0], self.kg.graph.number_of_nodes())
        self.assertEqual(embs.shape[1], 16)


class TestRAGModel(unittest.TestCase):

    def setUp(self):
        from rag_model import ClinicalRAGModel
        self.structured_dim = 40
        self.graph_emb_dim = 64
        self.latent_dim = 64
        self.batch = 8
        self.top_k = 3
        self.model = ClinicalRAGModel(
            structured_dim=self.structured_dim,
            graph_emb_dim=self.graph_emb_dim,
            latent_dim=self.latent_dim,
            hidden_dim=128,
            n_heads=4,
            n_tasks=3,
            top_k_retrieve=self.top_k,
        )

    def test_forward_shapes(self):
        sf = torch.randn(self.batch, self.structured_dim)
        ge = torch.randn(self.batch, self.graph_emb_dim)
        ctx = torch.randn(self.batch, self.top_k, self.latent_dim)
        out = self.model(sf, ge, ctx)
        self.assertEqual(out["logits"].shape, (self.batch, 3))
        self.assertEqual(out["risk_score"].shape, (self.batch,))
        self.assertIn("attention_weights", out)

    def test_no_nan_in_output(self):
        sf = torch.randn(4, self.structured_dim)
        ge = torch.randn(4, self.graph_emb_dim)
        ctx = torch.randn(4, self.top_k, self.latent_dim)
        out = self.model(sf, ge, ctx)
        self.assertFalse(torch.any(torch.isnan(out["logits"])))
        self.assertFalse(torch.any(torch.isnan(out["risk_score"])))


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.n = 100
        self.n_tasks = 3
        self.y_true = np.random.randint(0, 2, (self.n, self.n_tasks))
        self.y_prob = np.random.rand(self.n, self.n_tasks)

    def test_compute_metrics_keys(self):
        from evaluation import compute_metrics
        metrics = compute_metrics(self.y_true, self.y_prob)
        self.assertIn("mean_auroc", metrics)
        self.assertIn("mean_auprc", metrics)
        for task in ["readmission_30d", "adverse_event_90d", "mortality_1yr"]:
            self.assertIn(f"{task}_auroc", metrics)

    def test_ece(self):
        from evaluation import expected_calibration_error
        ece = expected_calibration_error(self.y_true[:, 0], self.y_prob[:, 0])
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_report_generation(self):
        from evaluation import generate_evaluation_report
        report = generate_evaluation_report(self.y_true, self.y_prob, [f"P{i}" for i in range(self.n)])
        self.assertIn("CORTEX-RAG", report)
        self.assertIn("AUROC", report.upper())


if __name__ == "__main__":
    unittest.main(verbosity=2)
