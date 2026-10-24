#!/usr/bin/env python3
"""Attention Heatmap Generator for Whole-Slide Diagnostic Images.

Generates high-resolution spatial heatmaps showing which morphological patterns
(confluent neoplastic sheets vs vascularized septa) drive the model's prediction
of osimertinib resistance and rapid progression.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def generate_spatial_attention_map(
    coords: np.ndarray,
    attention_weights: np.ndarray,
    grid_size: int = 100,
    blur_sigma: float = 1.5
) -> np.ndarray:
    """Project 1D patch attention weights onto a 2D spatial grid.
    
    Args:
        coords: Array of shape (N, 2) containing (x, y) patch centroid coordinates.
        attention_weights: Array of shape (N,) summing to 1.0.
        grid_size: Resolution of the output heatmap matrix.
        blur_sigma: Gaussian kernel smoothing factor.
    """
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
    
    # Normalize coordinates to [0, grid_size - 1]
    x_norm = np.clip(((coords[:, 0] - x_min) / max(x_max - x_min, 1e-6) * (grid_size - 1)).astype(int), 0, grid_size - 1)
    y_norm = np.clip(((coords[:, 1] - y_min) / max(y_max - y_min, 1e-6) * (grid_size - 1)).astype(int), 0, grid_size - 1)
    
    heatmap = np.zeros((grid_size, grid_size), dtype=float)
    
    # Scale attention to percentiles for high-contrast visualization
    scaled_attn = attention_weights / np.max(attention_weights)
    for i in range(len(coords)):
        heatmap[y_norm[i], x_norm[i]] += scaled_attn[i]
        
    # Gaussian smoothing
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(heatmap, sigma=blur_sigma)
    return smoothed


def plot_attention_overlay(
    heatmap: np.ndarray,
    patient_id: str,
    pfs_months: float,
    egfr_status: str,
    output_path: Path
):
    """Plot publication-quality attention heatmap overlay."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    im = ax.imshow(heatmap, cmap="inferno", origin="upper", aspect="equal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight (Vascular Decoupling Score)", fontsize=11, fontweight="bold")
    
    title_str = (
        f"InSituVision Spatial Attention Heatmap\n"
        f"Patient: {patient_id} | EGFR: {egfr_status} | PFS: {pfs_months:.1f} mo\n"
        f"High Attention (Bright) = Solid / Micropapillary Sanctuary Architecture"
    )
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Tissue X Coordinate (Normalized Bins)", fontsize=10)
    ax.set_ylabel("Tissue Y Coordinate (Normalized Bins)", fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Smoke test heatmap rendering
    np.random.seed(42)
    n = 250
    coords = np.random.uniform(100, 5000, size=(n, 2))
    attn = np.random.exponential(1.0, size=n)
    attn /= np.sum(attn)
    
    hmap = generate_spatial_attention_map(coords, attn)
    out_file = Path("/home/luca/Scrivania/spatial-he-vision-study4/results/figure_sample_attention_heatmap.png")
    plot_attention_overlay(hmap, "TCGA-05-4382", 10.97, "Mutant", out_file)
    print(f"Heatmap rendered successfully to {out_file}")
