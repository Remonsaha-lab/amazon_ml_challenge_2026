import pandas as pd
import numpy as np
import os
import re
from collections import Counter

DATA_DIR = os.path.join("student_resource", "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

print("=" * 80)
print("AMAZON ML CHALLENGE 2026: PHASE 1 DATASET FORENSICS & DIAGNOSTICS")
print("=" * 80)

# 1. File existence and sizes
print("\n--- 1. File Sizes & Record Counts ---")
train_files = {
    "train_source1": os.path.join(TRAIN_DIR, "train_source1.tsv"),
    "train_source2": os.path.join(TRAIN_DIR, "train_source2.tsv"),
    "train_source3": os.path.join(TRAIN_DIR, "train_source3.tsv"),
    "train_ground_truth": os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
}
test_files = {
    "test_source1": os.path.join(TEST_DIR, "test_source1.tsv"),
    "test_source2": os.path.join(TEST_DIR, "test_source2.tsv"),
    "test_source3": os.path.join(TEST_DIR, "test_source3.tsv"),
}

for name, path in {**train_files, **test_files}.items():
    sz_mb = os.path.getsize(path) / (1024 * 1024)
    # Count lines efficiently without loading entire file
    with open(path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f) - 1 # exclude header
    print(f"{name:20s}: {sz_mb:8.2f} MB | {line_count:,} records")

# 2. Inspect Train Ground Truth
print("\n--- 2. Ground Truth Cardinality & Match Analysis ---")
gt_path = train_files["train_ground_truth"]
df_gt = pd.read_csv(gt_path, sep="\t", dtype=str)
print(f"Total rows in train_ground_truth: {len(df_gt):,}")
print(f"Columns: {list(df_gt.columns)}")

# Handle NaN matched_entity_ids (singletons)
df_gt["matched_entity_ids"] = df_gt["matched_entity_ids"].fillna("")

def parse_matches(m_str):
    if not m_str or pd.isna(m_str) or m_str.strip() == "":
        return []
    return [x.strip() for x in m_str.split(",") if x.strip()]

df_gt["match_list"] = df_gt["matched_entity_ids"].apply(parse_matches)
df_gt["match_count"] = df_gt["match_list"].apply(len)

cardinality = Counter(df_gt["match_count"])
total_s1 = len(df_gt)
singletons = cardinality[0]
print(f"Total S1 entities in GT : {total_s1:,}")
print(f"Singletons (0 matches) : {singletons:,} ({singletons / total_s1 * 100:.2f}%)")
print(f"1 match                : {cardinality[1]:,} ({cardinality[1] / total_s1 * 100:.2f}%)")
print(f"2 matches              : {cardinality[2]:,} ({cardinality[2] / total_s1 * 100:.2f}%)")
print(f"3+ matches             : {sum(v for k, v in cardinality.items() if k >= 3):,} ({sum(v for k, v in cardinality.items() if k >= 3) / total_s1 * 100:.2f}%)")
print(f"Max matches for one S1 : {max(cardinality.keys())}")

# Distribution of match origins (S2 vs S3)
all_matches = [m for sublist in df_gt["match_list"] for m in sublist]
s2_count = sum(1 for m in all_matches if m.startswith("S2-"))
s3_count = sum(1 for m in all_matches if m.startswith("S3-"))
print(f"Total match pairs      : {len(all_matches):,}")
print(f"Matches from Source 2  : {s2_count:,} ({s2_count / len(all_matches) * 100:.2f}%)")
print(f"Matches from Source 3  : {s3_count:,} ({s3_count / len(all_matches) * 100:.2f}%)")

# Both S2 and S3 matched simultaneously?
def has_both(m_list):
    has_s2 = any(m.startswith("S2-") for m in m_list)
    has_s3 = any(m.startswith("S3-") for m in m_list)
    return has_s2 and has_s3

both_count = df_gt["match_list"].apply(has_both).sum()
print(f"S1 matching BOTH S2 & S3: {both_count:,} ({both_count / total_s1 * 100:.2f}%)")

# 3. Inspect Country Distributions
print("\n--- 3. Country Distributions ---")
for src_name in ["train_source1", "test_source1"]:
    path = train_files.get(src_name) or test_files.get(src_name)
    df_src = pd.read_csv(path, sep="\t", dtype=str, usecols=["country"])
    print(f"\nCountry counts in {src_name}:")
    print(df_src["country"].value_counts(dropna=False))

# 4. Check Nulls in Source 1, 2, 3
print("\n--- 4. Missing Values Inspection (Sample: train_source1) ---")
df_s1 = pd.read_csv(train_files["train_source1"], sep="\t", nrows=100000, dtype=str)
print("Null count in first 100k rows of train_source1:")
print(df_s1.isna().sum())
print("\nSample rows from train_source1:")
print(df_s1.head(5))

print("\n" + "=" * 80)
print("FORENSICS EXECUTION COMPLETE")
print("=" * 80)
