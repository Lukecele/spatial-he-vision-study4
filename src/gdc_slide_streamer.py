#!/usr/bin/env python3
"""GDC API slide streamer for TCGA-LUAD diagnostic Whole Slide Images (H&E).

Downloads diagnostic whole slide images (DX1) one at a time for streaming
in-memory processing without exhausting local disk quotas.
"""

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Generator, List, Optional


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"


def query_diagnostic_slides(patient_ids: List[str]) -> List[Dict]:
    """Query GDC for diagnostic SVS slide images belonging to given patient IDs."""
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": patient_ids
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": ["Slide Image"]
                }
            }
        ]
    }
    
    params = urllib.parse.urlencode({
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,cases.submitter_id",
        "format": "JSON",
        "size": str(min(len(patient_ids) * 5, 1000))
    })
    
    req = urllib.request.Request(
        f"{GDC_FILES_ENDPOINT}?{params}",
        headers={"User-Agent": "Mozilla/5.0 InSituVision-Client/1.0"}
    )
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode())
    
    hits = res.get("data", {}).get("hits", [])
    diagnostic_slides = []
    for hit in hits:
        name = hit.get("file_name", "")
        # Prefer diagnostic slides (DX) over tissue slides (TS/BS)
        if name.endswith(".svs") and ("DX" in name or "diagnostic" in name.lower()):
            submitter_id = hit["cases"][0]["submitter_id"] if hit.get("cases") else None
            diagnostic_slides.append({
                "patient_id": submitter_id,
                "file_id": hit["file_id"],
                "file_name": name,
                "file_size_mb": hit.get("file_size", 0) / (1024 * 1024)
            })
            
    return diagnostic_slides


def stream_download_slide(file_id: str, dest_path: Path) -> Path:
    """Download a single slide file from GDC API directly to destination path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GDC_DATA_ENDPOINT}/{file_id}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 InSituVision-Client/1.0"}
    )
    
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)  # 1 MB chunk
            if not chunk:
                break
            f.write(chunk)
            
    return dest_path


if __name__ == "__main__":
    # Test query for sample patient
    test_ids = ["TCGA-05-4382", "TCGA-05-4384"]
    slides = query_diagnostic_slides(test_ids)
    print(f"Discovered {len(slides)} diagnostic slides for {test_ids}:")
    for s in slides:
        print(f"  Patient: {s['patient_id']} | File: {s['file_name']} | Size: {s['file_size_mb']:.1f} MB | ID: {s['file_id']}")
