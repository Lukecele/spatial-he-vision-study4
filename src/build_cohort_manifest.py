#!/usr/bin/env python3
"""Build cohort manifest linking Study 3 patients to GDC diagnostic H&E slides."""

import json
import urllib.parse
import urllib.request
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STUDY3_DATA = Path("/home/luca/Scrivania/spatial-massive-validation-study3/data/combined_filtered_massive_cohort.csv")
OUTPUT_MANIFEST = ROOT / "data" / "cohort_manifest_study4.csv"

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"


def query_all_diagnostic_slides_for_luad():
    """Query GDC for all diagnostic (DX) slide images in TCGA-LUAD."""
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": ["TCGA-LUAD"]
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
    
    slides = {}
    page_size = 1000
    from_record = 0
    
    while True:
        params = urllib.parse.urlencode({
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,file_size,cases.submitter_id",
            "format": "JSON",
            "size": str(page_size),
            "from": str(from_record)
        })
        
        req = urllib.request.Request(
            f"{GDC_FILES_ENDPOINT}?{params}",
            headers={"User-Agent": "Mozilla/5.0 InSituVision-Client/1.0"}
        )
        
        with urllib.request.urlopen(req, timeout=60) as resp:
            res = json.loads(resp.read().decode())
            
        hits = res.get("data", {}).get("hits", [])
        if not hits:
            break
            
        for hit in hits:
            fname = hit.get("file_name", "")
            if fname.endswith(".svs") and ("DX" in fname or "diagnostic" in fname.lower()):
                cases = hit.get("cases", [])
                if cases:
                    pid = cases[0]["submitter_id"]
                    # If patient already has a slide, prefer DX1
                    if pid not in slides or "DX1" in fname:
                        slides[pid] = {
                            "gdc_file_id": hit["file_id"],
                            "file_name": fname,
                            "file_size_mb": round(hit.get("file_size", 0) / (1024 * 1024), 2)
                        }
                        
        from_record += page_size
        pagination = res.get("data", {}).get("pagination", {})
        if from_record >= pagination.get("total", 0):
            break
            
    return slides


def main():
    print(f"Loading Study 3 cohort from {STUDY3_DATA}...")
    df_study3 = pd.read_csv(STUDY3_DATA)
    tcga_df = df_study3[df_study3["cohort"] == "TCGA-LUAD"].copy()
    print(f"TCGA-LUAD patients in Study 3: {len(tcga_df)}")
    
    print("Querying GDC for diagnostic whole slide images...")
    slides_map = query_all_diagnostic_slides_for_luad()
    print(f"Discovered {len(slides_map)} diagnostic slides in TCGA-LUAD.")
    
    tcga_df["has_diagnostic_slide"] = tcga_df["patient_id"].map(lambda p: p in slides_map)
    tcga_df["gdc_file_id"] = tcga_df["patient_id"].map(lambda p: slides_map[p]["gdc_file_id"] if p in slides_map else "")
    tcga_df["slide_file_name"] = tcga_df["patient_id"].map(lambda p: slides_map[p]["file_name"] if p in slides_map else "")
    tcga_df["slide_size_mb"] = tcga_df["patient_id"].map(lambda p: slides_map[p]["file_size_mb"] if p in slides_map else 0.0)
    
    # Filter to patients with confirmed slides
    manifest_df = tcga_df[tcga_df["has_diagnostic_slide"]].copy()
    print(f"Manifest matches {len(manifest_df)} patients with diagnostic slides!")
    print(f"EGFR Status breakdown:\n{manifest_df['egfr_status'].value_counts()}")
    
    manifest_df.to_csv(OUTPUT_MANIFEST, index=False)
    print(f"Wrote cohort manifest to {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
