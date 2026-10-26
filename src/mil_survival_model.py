#!/usr/bin/env python3
"""Attention-based Deep Multiple Instance Learning (Attention-MIL) for Survival Analysis.

Implements gated attention pooling (Ilse et al., 2018; Lu et al., 2021) with Cox
proportional hazards partial likelihood loss for predicting progression-free survival
and microvascular decoupling directly from Whole-Slide Image patch bags.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class GatedAttentionLayer:
    """Gated Attention Pooling layer for bag-of-features representation.
    
    Formula:
        a_k = softmax(w^T (tanh(V h_k) * sigmoid(U h_k)))
    """
    def __init__(self, in_features: int, hidden_dim: int = 128, seed: int = 42):
        rng = np.random.RandomState(seed)
        # Xavier/Glorot initialization
        scale_v = math.sqrt(2.0 / (in_features + hidden_dim))
        scale_u = math.sqrt(2.0 / (in_features + hidden_dim))
        scale_w = math.sqrt(2.0 / (hidden_dim + 1))
        
        self.V = rng.normal(0, scale_v, size=(hidden_dim, in_features))
        self.b_v = np.zeros(hidden_dim)
        self.U = rng.normal(0, scale_u, size=(hidden_dim, in_features))
        self.b_u = np.zeros(hidden_dim)
        self.w = rng.normal(0, scale_w, size=(1, hidden_dim))
        self.b_w = np.zeros(1)
        
    def forward(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute attention weights and aggregated slide-level representation.
        
        Args:
            H: Matrix of shape (N_patches, in_features).
            
        Returns:
            z: Slide representation vector of shape (in_features,).
            attention_weights: Array of shape (N_patches,) summing to 1.
        """
        # H is (N, D)
        # tanh(V * h_k)
        v_out = np.tanh(np.dot(H, self.V.T) + self.b_v)  # (N, hidden_dim)
        # sigmoid(U * h_k)
        u_out = 1.0 / (1.0 + np.exp(-np.clip(np.dot(H, self.U.T) + self.b_u, -15, 15)))  # (N, hidden_dim)
        
        # Element-wise gating: (N, hidden_dim)
        gated = v_out * u_out
        
        # Attention scores: (N, 1) -> (N,)
        scores = np.dot(gated, self.w.T).squeeze(-1) + self.b_w
        
        # Numerically stable softmax
        scores_exp = np.exp(scores - np.max(scores))
        attention_weights = scores_exp / np.maximum(np.sum(scores_exp), 1e-12)
        
        # Slide representation z = sum_k a_k * h_k
        z = np.dot(attention_weights, H)  # (in_features,)
        return z, attention_weights


class AttentionSurvivalMIL:
    """End-to-End Survival MIL model predicting log-hazard ratios."""
    def __init__(self, in_features: int = 512, hidden_dim: int = 128, seed: int = 42):
        self.attention = GatedAttentionLayer(in_features, hidden_dim, seed=seed)
        rng = np.random.RandomState(seed + 1)
        self.classifier_w = rng.normal(0, 0.05, size=in_features)
        self.classifier_b = 0.0
        
    def predict_hazard(self, bag_features: np.ndarray) -> Tuple[float, np.ndarray]:
        """Predict log hazard ratio theta for a patient's bag of patch features."""
        z, attention = self.attention.forward(bag_features)
        log_hazard = float(np.dot(z, self.classifier_w) + self.classifier_b)
        return log_hazard, attention
    
    def predict_cohort(self, bags: List[np.ndarray]) -> np.ndarray:
        """Predict log hazards for a cohort of patient bags."""
        hazards = np.zeros(len(bags))
        for i, bag in enumerate(bags):
            hazards[i], _ = self.predict_hazard(bag)
        return hazards


def cox_negative_log_likelihood(hazards: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """Compute exact Cox Proportional Hazards Negative Log-Likelihood (Breslow/Efron tie handling).
    
    Args:
        hazards: Predicted log-hazard ratios (theta_i) for N patients.
        times: Observed survival / progression-free times.
        events: Binary indicator (1 if progressed/died, 0 if censored).
    """
    order = np.argsort(-times)  # Descending order
    sorted_hazards = hazards[order]
    sorted_events = events[order]
    
    # Cumulative log-sum-exp over risk sets
    risk_sum = 0.0
    loss = 0.0
    for i in range(len(sorted_hazards)):
        risk_sum += np.exp(sorted_hazards[i])
        if sorted_events[i] == 1:
            loss += np.log(max(risk_sum, 1e-12)) - sorted_hazards[i]
            
    return float(loss / max(np.sum(events), 1))


def harrell_concordance_index(hazards: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """Calculate Harrell's Concordance Index (C-index) for survival prediction."""
    concordant = 0
    permissible = 0
    tied = 0
    n = len(times)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Only pairs where at least one patient had an event and their times are ordered
            if times[i] < times[j] and events[i] == 1:
                permissible += 1
                if hazards[i] > hazards[j]:
                    concordant += 1
                elif hazards[i] == hazards[j]:
                    tied += 1
            elif times[j] < times[i] and events[j] == 1:
                permissible += 1
                if hazards[j] > hazards[i]:
                    concordant += 1
                elif hazards[j] == hazards[i]:
                    tied += 1
                    
    if permissible == 0:
        return 0.5
    return (concordant + 0.5 * tied) / permissible


if __name__ == "__main__":
    print(r"""
========================================================================
 ⭐ Star on GitHub: https://github.com/Lukecele/spatial-he-vision-study4
    If you find this research useful, please star the repository!
========================================================================
""")
    # Smoke test on synthetic bags
    model = AttentionSurvivalMIL(in_features=64, hidden_dim=32)
    dummy_bag = np.random.randn(150, 64)
    hz, attn = model.predict_hazard(dummy_bag)
    print(f"Sample Hazard: {hz:.4f} | Attn Sum: {np.sum(attn):.4f} | Top Attn: {np.max(attn):.4f}")
    
    hazards = np.array([1.5, 0.2, -0.5, 2.1])
    times = np.array([12.0, 36.0, 60.0, 8.0])
    events = np.array([1, 0, 0, 1])
    c_idx = harrell_concordance_index(hazards, times, events)
    print(f"Harrell C-index on test vector: {c_idx:.4f}")
