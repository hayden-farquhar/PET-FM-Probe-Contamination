# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PET-FM-Bench: Phase 2 Stage 1c — BiomedCLIP PMC Caption Scan
#
# **Runtime:** CPU | **Internet:** ON | **Time:** ~30-90 min | **GPU:** Not needed
#
# Pre-registration §4.6 (contamination tiering) Stage 1c: scan publicly
# available PubMed Central figure captions for explicit references to the
# evaluation patient IDs in PET-FM-Bench. Any caption mentioning a TCIA /
# AutoPET / HECKTOR patient code is a candidate contamination point — if the
# caption was in BiomedCLIP's training corpus, the model has seen that
# patient's data described in text and possibly an associated figure.
#
# **PMC-15M itself is not publicly downloadable.** Microsoft released BiomedCLIP
# weights but not the training corpus. We therefore scan an accessible
# **superset proxy** — the PMC Open Access (OA) subset, ~3.5M articles, all
# of which were eligible for inclusion in PMC-15M. This produces an *upper
# bound* audit per pre-reg §4.6 risk note: any patient ID found could
# plausibly have been in PMC-15M; absence is consistent with (but doesn't
# prove) absence from training.
#
# **Output:** `biomedclip_caption_matches.parquet` with schema
# `(fm, source_collection, patient_id, citation_doi, caption_excerpt)` —
# matches 07a/07b on the first three columns plus the audit-specific
# evidence columns.
#
# **VERIFY BEFORE FREEZE:** the regex patterns in `PATIENT_ID_REGEX` are seeded
# from the audit plan + canonical TCIA / AutoPET / HECKTOR ID formats.
# Cross-check against actual ID strings present in `task_splits.parquet`
# before uploading to OSF — false-negative regexes silently underestimate
# contamination.

# %% [markdown]
# ## 1. Setup

# %%
import gzip
import io
import json
import os
import re
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

!pip install -q requests lxml beautifulsoup4

import requests  # noqa: E402
from lxml import etree  # noqa: E402

# /tmp for raw PMC bulk downloads — large, ephemeral, not under the 20GB
# /kaggle/working limit (per kaggle_pipeline patterns memory).
TMP_DIR = Path("/tmp/pmc_oa")
TMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")

# %% [markdown]
# ## 2. Patient-ID regex patterns
#
# These match the canonical ID formats used by the TCIA collections that
# appear in PET-FM-Bench evaluation. Each pattern has a `source_collection`
# label so a hit is attributable.
#
# **VERIFY** by sampling actual patient IDs from `task_splits.parquet`:
# `pd.read_parquet('/kaggle/input/pet-fm-bench-task-splits/task_splits.parquet')`
# `.groupby('task')['patient_id'].first()` — confirm the regexes match real IDs.

# %%
PATIENT_ID_REGEX = [
    # NSCLC-Radiogenomics (Bakr et al.) — T4 evaluation cohort.
    # Two ID formats observed in FMCIB's nsclc_radiogenomics.csv:
    # `R01-005` (Stanford prefix) and `AMC-001` (Asan Medical Center prefix).
    {"pattern": r"\bR01-\d{3,4}\b",
     "source_collection": "NSCLC-Radiogenomics",
     "notes": "T4 evaluation cohort (Stanford R01- prefix)"},
    {"pattern": r"\bAMC-\d{3,4}\b",
     "source_collection": "NSCLC-Radiogenomics",
     "notes": "T4 evaluation cohort (AMC- prefix)"},

    # RIDER-Lung-PET-CT — T6 test-retest cohort.
    {"pattern": r"\bRIDER[\s_-]?\d{3,4}\b",
     "source_collection": "RIDER-Lung-PET-CT",
     "notes": "T6 test-retest cohort"},

    # ACRIN-NSCLC-FDG-PET — T7 outcome cohort.
    {"pattern": r"\bACRIN-NSCLC-FDG-PET-\d{3,4}\b",
     "source_collection": "ACRIN-NSCLC-FDG-PET",
     "notes": "T7 evaluation cohort"},

    # Lung-PET-CT-Dx — T8 subtype-classification cohort.
    {"pattern": r"\b(Lung_Dx|Lung-PET-CT-Dx)-[A-G]\d{3,4}\b",
     "source_collection": "Lung-PET-CT-Dx",
     "notes": "T8 evaluation cohort"},

    # Vienna QUADRA / T9 healthy controls.
    # NB: scanner-name "Biograph Vision Quadra" is a frequent string in PET
    # papers; the [\s_-] separator + trailing digits cuts most of those.
    {"pattern": r"\bQUADRA[\s_-]?\d{3,4}\b",
     "source_collection": "QUADRA",
     "notes": "T9 healthy-control cohort"},

    # NB: HECKTOR patient codes (CHGJ, CHUM, CHUS, MDA) were in earlier
    # versions of this audit but are intentionally REMOVED in v6:
    # - HECKTOR (T2/T3 in original registration) is NOT in the PET-FM-Bench
    #   v3 active task set (PROGRESS.md). Scanning for HECKTOR IDs adds
    #   false-positive risk without any contamination-detection upside.
    # - The MDA regex `\bMDA\d{3,4}\b` was a textbook false-positive trap:
    #   `MDA231` matched the MDA-MB-231 breast cancer cell line, one of the
    #   most-cited cell lines in cancer biology.
    # If HECKTOR is added back to the active task set in a future revision,
    # tighten patterns to require a dash (e.g., `\bMDA-\d{3,4}\b`).
]

# Compile once.
COMPILED_PATTERNS = [
    (re.compile(entry["pattern"]), entry["source_collection"], entry["notes"])
    for entry in PATIENT_ID_REGEX
]
print(f"Configured {len(COMPILED_PATTERNS)} patient-ID regex patterns")

# %% [markdown]
# ## 3. Acquire PMC OA caption corpus
#
# PMC OA bulk packages are at:
# `https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/`
# Each `.tar.gz` contains 1000s of full-text JATS XML articles. We extract
# `<fig><caption>` elements only — the audit doesn't need full body text.
#
# For first-pass scope, we download the most recent **monthly incremental**
# package(s) from before BiomedCLIP's training cutoff (March 2023, per
# Microsoft release notes). For a comprehensive audit, expand to the full
# baseline. Each package is ~1-2 GB.
#
# **Audit scope decision:**
# - `MODE = "sample"` (default): grab one or two recent packages + run
#   regex over them. Demonstrates the audit pipeline works end-to-end.
#   Suitable for the dry run / sanity-check phase.
# - `MODE = "full"`: iterate over every package on the FTP listing.
#   Required for the formal Phase 2 freeze artefact.

# %%
MODE = "sample"  # change to "full" for the formal Phase 2 freeze run

# **NCBI restructured PMC FTP in early 2026.** The canonical `oa_bulk/oa_comm/xml/`
# path was moved under `deprecated/` and the entire deprecated tree will be
# **removed in August 2026** (per `https://ftp.ncbi.nlm.nih.gov/pub/pmc/readme.txt`,
# updated 2026-04-10). After August 2026 the formal audit must migrate to
# the AWS S3 cloud service (`pmc-oa-opendata` bucket) — see
# `https://pmc.ncbi.nlm.nih.gov/tools/cloud/` for the canonical access docs.
# For now (verified 2026-04-27) the deprecated path is reachable and serves
# the same baseline + incremental tarballs.
PMC_OA_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_comm/xml/"

# **Verified 2026-04-27** against BiomedCLIP paper (Zhang et al. 2023,
# arXiv:2303.00915v3, Methods "Details of creating PMC-15M"):
#   "PubMed Central Open Access Subset (PMC-OA) contains 4.4 million publicly
#    available full-text articles (as of June 15, 2022). We download PMC-OA
#    from ncbi.nlm.nih.gov/pmc/tools/ftp/#indart..."
# Training corpus is bounded by the PMC-OA snapshot of 2022-06-15.
BIOMEDCLIP_TRAINING_CUTOFF = "2022-06"  # PMC-OA snapshot date per BiomedCLIP §Methods

# %% [markdown]
# ## 4. Discover PMC OA package list
#
# Parse the FTP HTML index to find package filenames + dates.

# %%
def list_pmc_oa_packages(base_url, cutoff_yyyymm):
    """Return baseline + pre-cutoff incremental packages.

    NCBI's PMC OA distribution has two package families:
      - **Baseline** (`oa_comm_xml.PMC<NNN>xxxxxx.baseline.YYYY-MM-DD.tar.gz`):
        a full archive snapshot, partitioned by PMCID range. The build-date
        in the filename is when the archive was rebuilt, NOT when the
        articles were published. Baseline packages from any build date
        contain articles spanning all years up to the build date — so for
        an upper-bound audit, baseline packages are ALWAYS in scope
        regardless of build date.
      - **Incremental** (`oa_comm_xml.incr.YYYY-MM-DD.tar.gz`): adds new
        articles published after the most-recent baseline rebuild.
        Incrementals dated AFTER `cutoff_yyyymm` contain articles BiomedCLIP
        could not have seen, so we filter them out.
    """
    print(f"Fetching listing: {base_url}")
    try:
        resp = requests.get(base_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  FAILED to fetch listing: {e}")
        return []

    baselines, incrementals_in_scope = [], []
    for line in resp.text.splitlines():
        m = re.search(r'href="(oa_comm_xml\.[^"]+\.tar\.gz)"', line)
        if not m:
            continue
        fname = m.group(1)
        if ".baseline." in fname:
            baselines.append(fname)
        elif ".incr." in fname:
            date_match = re.search(r"\.incr\.(\d{4}-\d{2})", fname)
            if date_match and date_match.group(1) <= cutoff_yyyymm:
                incrementals_in_scope.append(fname)

    print(f"  baseline packages found: {len(baselines)} (always in scope)")
    print(f"  incremental packages dated ≤ {cutoff_yyyymm}: {len(incrementals_in_scope)}")
    return baselines, incrementals_in_scope


baselines, incrementals = list_pmc_oa_packages(PMC_OA_BASE, BIOMEDCLIP_TRAINING_CUTOFF)

if MODE == "sample":
    # One small baseline shard + zero incrementals = enough to validate the
    # streaming + regex pipeline without spending hours on download.
    packages = baselines[:1] if baselines else []
    print(f"  MODE=sample → scanning 1 baseline package "
          f"({packages[0] if packages else 'none'})")
elif MODE == "full":
    packages = baselines + incrementals
    print(f"  MODE=full → scanning {len(baselines)} baselines + "
          f"{len(incrementals)} pre-cutoff incrementals = "
          f"{len(packages)} total packages")
else:
    packages = []
    print(f"  ✗ Unknown MODE={MODE}; expected 'sample' or 'full'")

# %% [markdown]
# ## 5. Stream each package, extract figure captions, regex-scan
#
# Memory-friendly approach: open each tar.gz as a stream, parse member XML
# files one at a time, extract `<fig><caption>` text only, scan, drop.

# %%
def extract_captions_and_doi(xml_bytes):
    """Parse a JATS XML article; yield (caption_text, doi)."""
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return

    # DOI extraction
    doi_elem = root.find(".//article-id[@pub-id-type='doi']")
    doi = doi_elem.text.strip() if (doi_elem is not None and doi_elem.text) else ""

    # Figure captions
    for fig in root.findall(".//fig"):
        caption = fig.find(".//caption")
        if caption is None:
            continue
        text = " ".join(caption.itertext()).strip()
        if text:
            yield text, doi


def scan_caption(caption_text, doi):
    """Apply all patient-ID regexes; return list of match records."""
    matches = []
    for pattern, src_collection, notes in COMPILED_PATTERNS:
        for m in pattern.finditer(caption_text):
            pid = m.group(0)
            # Extract a ±60-char excerpt around the match for evidence.
            start = max(0, m.start() - 60)
            end = min(len(caption_text), m.end() + 60)
            excerpt = caption_text[start:end].replace("\n", " ").strip()
            matches.append({
                "fm": "biomedclip",
                "source_collection": src_collection,
                "patient_id": pid,
                "citation_doi": doi,
                "caption_excerpt": excerpt,
                "evidence": notes,
            })
    return matches


def download(url, dest):
    """Streaming download; resume-friendly for large packages."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    cached: {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return
    print(f"    downloading: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=2**20):
                f.write(chunk)
    print(f"    saved: {dest.stat().st_size/1e6:.0f} MB")


all_matches = []
package_summary = []

for pkg in packages:
    pkg_url = PMC_OA_BASE + pkg
    pkg_path = TMP_DIR / pkg
    print(f"\n=== {pkg} ===")

    try:
        download(pkg_url, pkg_path)
    except Exception as e:
        print(f"  download failed: {e}")
        package_summary.append({"package": pkg, "n_articles": 0,
                                "n_matches": 0, "status": "DOWNLOAD_FAILED"})
        continue

    n_articles = 0
    n_matches_pkg = 0
    try:
        with tarfile.open(pkg_path, "r:gz") as tar:
            for member in tar:
                if not member.name.endswith(".xml"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                xml_bytes = f.read()
                n_articles += 1
                for caption_text, doi in extract_captions_and_doi(xml_bytes):
                    hits = scan_caption(caption_text, doi)
                    if hits:
                        all_matches.extend(hits)
                        n_matches_pkg += len(hits)
                if n_articles % 5000 == 0:
                    print(f"    ...{n_articles} articles scanned, "
                          f"{n_matches_pkg} matches so far")
    except Exception as e:
        print(f"  scan failed mid-package: {type(e).__name__}: {e}")
        package_summary.append({"package": pkg, "n_articles": n_articles,
                                "n_matches": n_matches_pkg, "status": "SCAN_PARTIAL"})
        continue

    print(f"  {n_articles} articles, {n_matches_pkg} matches")
    package_summary.append({"package": pkg, "n_articles": n_articles,
                            "n_matches": n_matches_pkg, "status": "OK"})

    # Free disk: large packages eat /tmp space quickly.
    try:
        pkg_path.unlink()
    except FileNotFoundError:
        pass

# %% [markdown]
# ## 6. Save matches

# %%
if all_matches:
    matches_df = pd.DataFrame(all_matches).drop_duplicates(
        subset=["fm", "source_collection", "patient_id", "citation_doi"]
    )
else:
    # Empty but well-typed dataframe for downstream concat compatibility.
    matches_df = pd.DataFrame(columns=[
        "fm", "source_collection", "patient_id",
        "citation_doi", "caption_excerpt", "evidence",
    ])

summary_df = pd.DataFrame(package_summary)

print(f"\n=== BiomedCLIP caption-scan summary ===")
print(f"Mode:                          {MODE}")
print(f"Packages scanned:              {len(summary_df)}")
print(f"Total articles scanned:        {summary_df['n_articles'].sum() if len(summary_df) else 0}")
print(f"Unique (collection, pid) hits: "
      f"{matches_df.drop_duplicates(['source_collection', 'patient_id']).shape[0]}")
print(f"Total caption matches:         {len(matches_df)}")

if len(summary_df):
    print("\n" + summary_df.to_string(index=False))

if len(matches_df):
    print("\nTop hits by collection:")
    print(matches_df.groupby("source_collection")["patient_id"]
          .nunique().sort_values(ascending=False).to_string())

manifest_path = OUT_DIR / "biomedclip_caption_matches.parquet"
summary_path = OUT_DIR / "biomedclip_scan_summary.csv"
metadata_path = OUT_DIR / "biomedclip_manifest_metadata.json"

matches_df.to_parquet(manifest_path, index=False)
summary_df.to_csv(summary_path, index=False)

with open(metadata_path, "w") as f:
    json.dump({
        "fm": "biomedclip",
        "freeze_timestamp_utc": freeze_timestamp,
        "source_paper": "Zhang et al. 2023, arXiv:2303.00915v3 "
                        "(BiomedCLIP / PMC-15M)",
        "source_corpus": "PubMed Central Open Access Subset (PMC-OA)",
        "source_corpus_url": "https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/",
        "biomedclip_training_cutoff": BIOMEDCLIP_TRAINING_CUTOFF,
        "biomedclip_training_cutoff_source": (
            "Zhang et al. 2023 §Methods: 'PMC-OA contains 4.4 million "
            "publicly available full-text articles (as of June 15, 2022)'"
        ),
        "audit_mode": MODE,
        "audit_type": "upper_bound",
        "audit_caveat": (
            "PMC-15M itself is not publicly downloadable (Microsoft released "
            "BiomedCLIP weights but not the training corpus). PMC-OA is the "
            "accessible superset that BiomedCLIP extracted PMC-15M from; "
            "matches here are an upper bound on what BiomedCLIP could have "
            "seen. Absence is consistent with — but does not prove — absence "
            "from training."
        ),
        "verification_status": (
            "VERIFIED 2026-04-27 against BiomedCLIP paper §Methods 'Details "
            "of creating PMC-15M'. Cutoff date and source URL exact."
        ),
        "n_packages_scanned": len(summary_df),
        "n_articles_scanned": int(summary_df["n_articles"].sum()) if len(summary_df) else 0,
        "n_unique_patient_hits": int(matches_df.drop_duplicates(
            ["source_collection", "patient_id"]).shape[0]),
        "n_total_matches": int(len(matches_df)),
        "verify_before_freeze": (
            "PATIENT_ID_REGEX patterns seeded from audit plan. Cross-check "
            "against actual IDs in task_splits.parquet before formal run. "
            "Set MODE='full' for Phase 2 freeze run (sample mode is for the "
            "dry-run pipeline check only)."
        ),
    }, f, indent=2)

print(f"\nWrote: {manifest_path}")
print(f"Wrote: {summary_path}")
print(f"Wrote: {metadata_path}")

# %% [markdown]
# ## 7. Done
#
# Commit with **"Save & Run All"** to persist as
# `pet-fm-bench-biomedclip-manifest`. For the formal Phase 2 freeze, set
# `MODE = "full"` above and re-run (expect 6-12 hours wall time on Kaggle CPU
# vs ~30-60 min in sample mode).
