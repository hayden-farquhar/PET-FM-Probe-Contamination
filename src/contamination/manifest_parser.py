"""Parse FM training manifests and compute patient-ID intersection with benchmark splits.

Pre-registration Section 4: Contamination Audit Protocol.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# Contamination confidence tiers (Section 4.3)
TIER_CONFIRMED = 1    # Study-UID-level match
TIER_PROBABLE = 2     # Patient-ID match, same collection/modality
TIER_POSSIBLE = 3     # Patient-ID match, different collection/modality
TIER_PROXY = 4        # Caption-scan or institutional overlap
TIER_CLEAN = 0        # FM domain categorically excludes benchmark modality


def compute_patient_overlap(
    bench_patient_ids: list[str],
    fm_patient_ids: list[str],
) -> dict:
    """Compute patient-ID intersection between benchmark and FM training set.

    Args:
        bench_patient_ids: Patient IDs in the benchmark evaluation split.
        fm_patient_ids: Patient IDs in the FM's training manifest.

    Returns:
        dict with intersection statistics.
    """
    bench_set = set(bench_patient_ids)
    fm_set = set(fm_patient_ids)
    intersection = bench_set & fm_set

    return {
        "patients_in_bench": len(bench_set),
        "patients_in_fm_training": len(fm_set),
        "intersection_count": len(intersection),
        "intersection_fraction": len(intersection) / len(bench_set) if bench_set else 0.0,
        "intersection_ids": sorted(intersection),
    }


def audit_ct_fm(
    bench_manifest: pd.DataFrame,
    ct_fm_collection_list: list[str],
    bench_collection: str,
    patient_id_col: str = "patient_id",
) -> dict:
    """Audit CT-FM (FMCIB / CT-FM) contamination via TCIA collection overlap.

    CT-FM publishes a list of TCIA collections used for training.
    If the benchmark task's source collection appears in that list,
    contamination is possible at the collection level (Tier 2-3).

    For patient-level audit (Tier 1-2), the FM's per-patient training
    manifest would be needed — not always available.
    """
    collection_overlap = bench_collection in ct_fm_collection_list

    result = {
        "fm": "ct_fm",
        "task_collection": bench_collection,
        "collection_in_fm_training": collection_overlap,
        "confidence_tier": TIER_PROBABLE if collection_overlap else TIER_CLEAN,
        "audit_method": "collection_list_match",
    }

    return result


def audit_declared_clean(fm_name: str, reason: str) -> dict:
    """For FMs whose training domain categorically excludes PET.

    Used for: Merlin (Stanford CT only), RAD-DINO (CXR only), DINOv2 (ImageNet).
    """
    return {
        "fm": fm_name,
        "confidence_tier": TIER_CLEAN,
        "contamination_fraction": 0.0,
        "audit_method": "declared_clean",
        "reason": reason,
    }


def audit_biomedclip_caption_scan(
    caption_sample: list[str],
    search_terms: list[str],
) -> dict:
    """Proxy audit for BiomedCLIP via PMC-15M caption keyword scan.

    Pre-registration Section 4.2: draw 100k caption sample, apply regex
    for TCIA collection names, AutoPET identifiers, etc.

    Args:
        caption_sample: List of caption strings from PMC-15M.
        search_terms: List of search terms (collection names, dataset IDs).

    Returns:
        dict with keyword hit rates per search term.
    """
    import re

    results = {}
    total = len(caption_sample)

    for term in search_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        hits = sum(1 for cap in caption_sample if pattern.search(cap))
        results[term] = {
            "hits": hits,
            "total": total,
            "hit_rate": hits / total if total > 0 else 0.0,
        }

    return {
        "fm": "biomedclip",
        "confidence_tier": TIER_PROXY,
        "audit_method": "caption_scan",
        "search_results": results,
        "overall_hit_rate": sum(r["hits"] for r in results.values()) / total if total > 0 else 0.0,
    }


def build_contamination_matrix(
    fm_audits: list[dict],
    tasks: list[str],
) -> pd.DataFrame:
    """Build the (FM × task) contamination matrix.

    Output schema per registration Section 4.6:
    [fm, task, split, patients_in_bench, patients_in_fm_training,
     intersection_count, intersection_fraction, audit_method,
     confidence_tier, unauditable_fraction]
    """
    rows = []
    for audit in fm_audits:
        for task in tasks:
            rows.append({
                "fm": audit.get("fm"),
                "task": task,
                "confidence_tier": audit.get("confidence_tier"),
                "contamination_fraction": audit.get("contamination_fraction", 0.0),
                "audit_method": audit.get("audit_method"),
            })

    return pd.DataFrame(rows)


# FMCIB and CT-FM known TCIA training collections (from published manifests)
# These will be populated when we parse the actual manifests
FMCIB_TCIA_COLLECTIONS = [
    # To be filled from AIM-Harvard/foundation-cancer-image-biomarker docs
]

CT_FM_TCIA_COLLECTIONS = [
    # To be filled from project-lighter/CT-FM docs
]

# Declared-clean FMs
CLEAN_FMS = {
    "merlin": "Pre-trained exclusively on Stanford institutional CT. No TCIA or public PET data.",
    "rad_dino": "Pre-trained on 5 public CXR datasets (MIMIC-CXR, CheXpert, NIH ChestX-ray14, PadChest, BRAX). No PET.",
    "dinov2": "Pre-trained on LVD-142M / ImageNet-22k. No medical imaging data.",
    "random_init": "No pre-training. Random weights.",
}
