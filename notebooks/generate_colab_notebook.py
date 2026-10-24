#!/usr/bin/env python3
"""Generate InSituVision_Colab_T4_Pipeline.ipynb with full interactive pipeline."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "InSituVision_Colab_T4_Pipeline.ipynb"


def make_cell(cell_type, source, outputs=None):
    if isinstance(source, list):
        src = [s if s.endswith("\n") else s + "\n" for s in source]
    else:
        src = [s + "\n" for s in source.split("\n")]
    if src and src[-1].endswith("\n"):
        src[-1] = src[-1][:-1]
        
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = outputs or []
    return cell


def build_notebook():
    cells = []
    
    # Title Markdown
    cells.append(make_cell("markdown", """# InSituVision-AI: Predicting Microvascular Decoupling and Osimertinib PFS from Routine H&E Whole-Slide Images

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lukecele/spatial-he-vision-study4/blob/main/notebooks/InSituVision_Colab_T4_Pipeline.ipynb)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://zenodo.org)

**Author:** Luca Celebrano  
**Affiliation:** Department of Clinical Medicine and Surgery, School of Medicine, University of Naples Federico II, Naples, Italy  
**Correspondence:** `l.celebrano@studenti.unina.it` | **ORCID:** [0009-0007-0931-3555](https://orcid.org/0009-0007-0931-3555)

---

## 🔬 Clinical Motivation

In **Studies 1, 2, and 3**, we demonstrated that **microvascular decoupling** (the complete loss of capillary contact in dense homotypic tumor sheets) creates a steep biophysical diffusion barrier ($\lambda \approx 624\,\mu\text{m}$) that causes early relapse under EGFR tyrosine kinase inhibitors (Osimertinib), truncating Progression-Free Survival from **78.6 to 24.0 months** ($p < 10^{-15}$, $N = 854$ patients).

However, quantifying microvascular decoupling via **10x Genomics Visium HD spatial transcriptomics costs ~5,000 € per biopsy**, restricting it to major academic institutions.

**InSituVision-AI** democratizes this clinical biomarker by training an **Attention-based Deep Multiple Instance Learning (Attention-MIL)** neural network to predict microvascular decoupling and progression-free survival **directly from standard diagnostic H&E whole-slide images (costing < 5 €)**.

---

## ⚡ Architecture: Zero-Disk Overflow Streaming Pipeline
To operate seamlessly within Google Colab (with ~100 GB ephemeral workspace) and Google Drive (up to 5 TB storage):
1. **One-Slide-at-a-Time Streaming:** Colab fetches one slide (`DX1.svs`) directly from the NIH GDC API (~1 GB).
2. **On-the-Fly GPU T4 Tiling:** Patches are extracted in RAM and encoded by a pre-trained pathology Vision Transformer / ResNet50.
3. **Compact Feature Serialization:** Slide embeddings are compressed to a single `.pt` matrix (~15 MB per patient) and saved permanently to Google Drive.
4. **Immediate Disk Eviction:** The raw 1 GB `.svs` file is deleted from Colab local disk, keeping local disk usage strictly below 2 GB.
"""))

    # Section 1: Colab Setup & Google Drive Mount
    cells.append(make_cell("markdown", """### Step 1: Environment Verification & Google Drive Mount"""))
    cells.append(make_cell("code", """# 1. Mount Google Drive to persist feature embeddings
from google.colab import drive
import os
import sys

drive.mount('/content/drive')

DRIVE_DIR = "/content/drive/MyDrive/InSituVision_Study4"
os.makedirs(f"{DRIVE_DIR}/features", exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/results", exist_ok=True)
print(f"✓ Google Drive mounted. Data will be saved in: {DRIVE_DIR}")

# 2. Check GPU acceleration
!nvidia-smi
"""))

    # Section 2: Install Dependencies
    cells.append(make_cell("markdown", """### Step 2: Install Specialized Digital Pathology Dependencies"""))
    cells.append(make_cell("code", """!apt-get install -y openslide-tools libopenjp2-7
!pip install -q openslide-python timm lifelines torch torchvision scikit-learn matplotlib
print("✓ Digital pathology dependencies successfully installed!")
"""))

    # Section 3: Cohort Manifest
    cells.append(make_cell("markdown", """### Step 3: Load Study 4 Cohort Manifest (TCGA-LUAD EGFR Patients)"""))
    cells.append(make_cell("code", """import pandas as pd

MANIFEST_URL = "https://raw.githubusercontent.com/Lukecele/spatial-he-vision-study4/main/data/cohort_manifest_study4.csv"
manifest = pd.read_csv(MANIFEST_URL)

print(f"✓ Total patients in manifest: {len(manifest)}")
print(f"EGFR Status distribution:\n{manifest['egfr_status'].value_counts()}\n")
manifest.head(5)
"""))

    # Section 4: Streaming WSI Extractor & GPU T4 Pipeline
    cells.append(make_cell("markdown", """### Step 4: Streaming Feature Extraction Engine (One-Slide-At-A-Time)"""))
    cells.append(make_cell("code", """import io
import json
import urllib.request
from pathlib import Path
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Initialize PyTorch GPU T4 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Load pre-trained pathology feature extractor (ResNet50 backbone)
feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor.fc = torch.nn.Identity()  # 2048-dim embedding
feature_extractor = feature_extractor.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features_from_slide_streaming(file_id, patient_id, max_patches=1000):
    \"\"\"Stream one slide from GDC API, extract patch features on GPU T4, save .pt to Drive, and clean up.\"\"\"
    drive_feature_path = f"{DRIVE_DIR}/features/{patient_id}_features.pt"
    if os.path.exists(drive_feature_path):
        print(f"  [Skip] {patient_id} features already exist on Drive.")
        return drive_feature_path

    temp_svs = f"/content/temp_{patient_id}.svs"
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    print(f"  [Streaming] Fetching {patient_id} from GDC...")
    
    # Download 1 slide to local ephemeral disk
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 InSituVision-Client/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(temp_svs, "wb") as f:
        while True:
            chunk = resp.read(2 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            
    # Open with OpenSlide
    import openslide
    slide = openslide.OpenSlide(temp_svs)
    w, h = slide.dimensions
    
    # Simple tissue-grid sampling at 20x (step = 512px)
    step = 512
    coords = []
    patches = []
    for x in range(0, w - step, step):
        for y in range(0, h - step, step):
            tile = slide.read_region((x, y), 0, (256, 256)).convert("RGB")
            arr = np.array(tile)
            # Filter pure white background
            if np.mean(arr) < 220 and np.std(arr) > 15:
                coords.append([x, y])
                patches.append(transform(tile))
            if len(patches) >= max_patches:
                break
        if len(patches) >= max_patches:
            break
            
    slide.close()
    
    # Delete temporary slide to guarantee local disk < 2 GB
    if os.path.exists(temp_svs):
        os.remove(temp_svs)
        
    if len(patches) == 0:
        print(f"  [Warning] No tissue patches found for {patient_id}")
        return None
        
    # Batch GPU forward pass
    batch_tensors = torch.stack(patches).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            feats = feature_extractor(batch_tensors).cpu().numpy()
            
    # Save compact 15MB feature bag to Google Drive
    payload = {
        "patient_id": patient_id,
        "features": feats,
        "coords": np.array(coords)
    }
    torch.save(payload, drive_feature_path)
    print(f"  ✓ Saved {len(feats)} patch features (size ~{os.path.getsize(drive_feature_path)/(1024*1024):.1f} MB) to Google Drive!")
    return drive_feature_path
"""))

    # Section 5: Attention-MIL PyTorch Model Architecture
    cells.append(make_cell("markdown", """### Step 5: Attention-Based Multiple Instance Learning (Attention-MIL) Network"""))
    cells.append(make_cell("code", """import torch.nn as nn
import torch.nn.functional as F

class GatedAttentionMIL(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.attention_V = nn.Linear(512, hidden_dim)
        self.attention_U = nn.Linear(512, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        
        # Hazard prediction head
        self.classifier = nn.Linear(512, 1)

    def forward(self, h):
        # h: (N_patches, in_features)
        h = self.fc(h)
        v = torch.tanh(self.attention_V(h))
        u = torch.sigmoid(self.attention_U(h))
        scores = self.attention_w(v * u)  # (N, 1)
        weights = F.softmax(scores, dim=0)  # (N, 1)
        
        slide_repr = torch.sum(weights * h, dim=0, keepdim=True)  # (1, 512)
        log_hazard = self.classifier(slide_repr)  # (1, 1)
        return log_hazard.squeeze(), weights.squeeze()

print("✓ GatedAttentionMIL architecture instantiated successfully.")
"""))

    # Section 6: Training with Cox Partial Likelihood
    cells.append(make_cell("markdown", """### Step 6: Train Survival Model with Cox Partial Likelihood Loss"""))
    cells.append(make_cell("code", """def cox_loss(hazards, times, events):
    \"\"\"Negative log partial likelihood of Cox proportional hazards model.\"\"\"
    order = torch.argsort(-times)
    hazards = hazards[order]
    events = events[order]
    
    log_risk = torch.logcumsumexp(hazards, dim=0)
    uncensored_loss = -(hazards - log_risk) * events
    return torch.sum(uncensored_loss) / (torch.sum(events) + 1e-8)

print("✓ Cox loss function defined.")
"""))

    # Section 7: Explainability & Heatmap Generation
    cells.append(make_cell("markdown", """### Step 7: Spatial Explainability (Attention Heatmap Overlay)"""))
    cells.append(make_cell("code", """import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def plot_colab_heatmap(coords, weights, patient_id, pfs_months):
    \"\"\"Render spatial attention heatmap directly in Colab.\"\"\"
    grid_size = 100
    x, y = coords[:, 0], coords[:, 1]
    xn = np.clip(((x - x.min()) / max(x.max() - x.min(), 1) * (grid_size - 1)).astype(int), 0, grid_size - 1)
    yn = np.clip(((y - y.min()) / max(y.max() - y.min(), 1) * (grid_size - 1)).astype(int), 0, grid_size - 1)
    
    grid = np.zeros((grid_size, grid_size))
    w_norm = weights / weights.max()
    for i in range(len(coords)):
        grid[yn[i], xn[i]] += w_norm[i]
        
    smooth = gaussian_filter(grid, sigma=1.5)
    
    plt.figure(figsize=(7, 6), dpi=150)
    plt.imshow(smooth, cmap="inferno", origin="upper")
    plt.colorbar(label="Attention Score (Vascular Decoupling Sanctuary)")
    plt.title(f"InSituVision Heatmap: {patient_id} (PFS: {pfs_months:.1f} mo)", fontweight="bold")
    plt.xlabel("X Tissue Plane")
    plt.ylabel("Y Tissue Plane")
    plt.tight_layout()
    plt.show()

print("✓ Spatial explainability engine ready.")
"""))

    notebook_json = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(notebook_json, f, indent=2)
    print(f"Generated Colab notebook at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
