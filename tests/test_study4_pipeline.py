#!/usr/bin/env python3
"""Automated unit test suite for Study 4 (InSituVision-AI)."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mil_survival_model import GatedAttentionLayer, AttentionSurvivalMIL, cox_negative_log_likelihood, harrell_concordance_index
from attention_heatmap_generator import generate_spatial_attention_map


class TestStudy4Pipeline(unittest.TestCase):
    def test_cohort_manifest_integrity(self):
        manifest_path = ROOT / "data" / "cohort_manifest_study4.csv"
        self.assertTrue(manifest_path.exists(), f"Missing manifest: {manifest_path}")
        
        df = pd.read_csv(manifest_path)
        self.assertGreaterEqual(len(df), 500)
        
        required_cols = [
            "patient_id", "cohort", "histology", "egfr_status",
            "follow_up_months", "progression_event", "os_months", "os_event",
            "gdc_file_id", "slide_file_name", "slide_size_mb"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)
            
        # Check EGFR-mutant cases
        egfr_mut = df[df["egfr_status"] == "Mutant"]
        self.assertGreaterEqual(len(egfr_mut), 70)
        self.assertTrue(all(egfr_mut["gdc_file_id"].str.len() > 10))

    def test_gated_attention_pooling(self):
        layer = GatedAttentionLayer(in_features=64, hidden_dim=32, seed=123)
        h_bag = np.random.randn(100, 64)
        z, attn = layer.forward(h_bag)
        
        self.assertEqual(z.shape, (64,))
        self.assertEqual(attn.shape, (100,))
        self.assertAlmostEqual(float(np.sum(attn)), 1.0, places=5)
        self.assertTrue(np.all(attn >= 0.0))

    def test_cox_loss_and_concordance(self):
        hazards = np.array([2.5, 1.2, 0.0, -1.5])
        times = np.array([5.0, 15.0, 30.0, 50.0])
        events = np.array([1, 1, 0, 0])
        
        loss = cox_negative_log_likelihood(hazards, times, events)
        self.assertGreater(loss, 0.0)
        self.assertTrue(np.isfinite(loss))
        
        c_index = harrell_concordance_index(hazards, times, events)
        self.assertEqual(c_index, 1.0)

    def test_spatial_attention_heatmap(self):
        coords = np.array([[10, 10], [50, 50], [90, 90]])
        attn = np.array([0.1, 0.6, 0.3])
        grid = generate_spatial_attention_map(coords, attn, grid_size=50)
        
        self.assertEqual(grid.shape, (50, 50))
        self.assertTrue(np.all(grid >= 0.0))
        self.assertTrue(np.all(np.isfinite(grid)))
        self.assertGreater(np.max(grid), 0.0)


if __name__ == "__main__":
    unittest.main()
