"""
ML Challenge 2026: Automated Submission Packaging Utility.
Builds the final submission.zip containing:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv
  - code/business_entity_resolution/ (src/, requirements.txt, README.md)
  - Documentation.md
"""

import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_ZIP = ROOT_DIR / "submission.zip"

print(f"Creating submission archive: {OUTPUT_ZIP}")

files_to_pack = [
    ("output/matching_results.tsv", "output/matching_results.tsv"),
    ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
    ("student_resource/Documentation_template.md", "Documentation.md"),
    ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
]

# Source python files
src_dir = ROOT_DIR / "code" / "business_entity_resolution" / "src"
for p in src_dir.glob("*.py"):
    rel = p.relative_to(ROOT_DIR)
    files_to_pack.append((str(rel), str(rel)))

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for src_path, arc_name in files_to_pack:
        full_src = ROOT_DIR / src_path
        if full_src.exists():
            zf.write(full_src, arcname=arc_name)
            print(f"  + Added: {arc_name} ({full_src.stat().st_size:,} bytes)")
        else:
            print(f"  ! Warning: {src_path} not found")

print(f"\nSuccessfully created {OUTPUT_ZIP.name} ({OUTPUT_ZIP.stat().st_size / (1024*1024):.2f} MB)")
