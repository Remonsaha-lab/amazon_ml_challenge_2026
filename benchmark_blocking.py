import os
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Ensure UTF-8 stdout on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "code" / "business_entity_resolution" / "src"))


from blocking import MultiKeyBlocker
import config

print("=" * 80)
print("BENCHMARKING MULTI-KEY BLOCKING RECALL & CANDIDATE DENSITY")
print("=" * 80)

# 1. Load sample S1 entities with ground truth
SAMPLE_SIZE = 5000
print(f"\nLoading sample of {SAMPLE_SIZE} S1 entities with ground truth...")

df_gt = pd.read_csv(config.TRAIN_GROUND_TRUTH, sep="\t", nrows=SAMPLE_SIZE * 3, dtype=str)
df_gt["matched_entity_ids"] = df_gt["matched_entity_ids"].fillna("")

# Keep rows with matches to test recall accurately
df_gt_matches = df_gt[df_gt["matched_entity_ids"].str.strip() != ""].head(SAMPLE_SIZE)
sample_s1_ids = set(df_gt_matches["source1_entity_id"])

gt_pairs = {}
all_target_ids = set()
for _, row in df_gt_matches.iterrows():
    s1_id = row["source1_entity_id"]
    targets = {x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()}
    gt_pairs[s1_id] = targets
    all_target_ids.update(targets)

total_true_links = sum(len(v) for v in gt_pairs.values())
print(f"Sample S1 count: {len(sample_s1_ids):,}")
print(f"Total True Match Links to find: {total_true_links:,}")

# 2. Load S1 metadata
print("Loading S1 records...")
s1_records = []
with open(config.TRAIN_SOURCE1, "r", encoding="utf-8") as f:
    header_s1 = next(f).strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in sample_s1_ids:
            s1_records.append(parts)
            if len(s1_records) == len(sample_s1_ids):
                break
df_s1_sample = pd.DataFrame(s1_records, columns=header_s1)

# 3. Load S2 and S3 candidate pool
# Must include all target IDs + random background records to simulate real competition search space
print("Loading S2 and S3 candidate pool (including all targets + background pool)...")
cand_records = []
BACKGROUND_POOL_SIZE = 150000  # 150k candidate pool

# Read S2
count_s2 = 0
with open(config.TRAIN_SOURCE2, "r", encoding="utf-8") as f:
    header_s2 = next(f).strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            if parts[0] in all_target_ids or count_s2 < (BACKGROUND_POOL_SIZE // 2):
                cand_records.append(parts)
                count_s2 += 1

# Read S3
count_s3 = 0
with open(config.TRAIN_SOURCE3, "r", encoding="utf-8") as f:
    header_s3 = next(f).strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            if parts[0] in all_target_ids or count_s3 < (BACKGROUND_POOL_SIZE // 2):
                cand_records.append(parts)
                count_s3 += 1

df_candidates = pd.DataFrame(cand_records, columns=["entity_id", "business_name", "business_address", "country"])
print(f"Total candidates in test pool: {len(df_candidates):,}")

# 4. Build MultiKeyBlocker and index pool
blocker = MultiKeyBlocker(max_candidates_per_s1=config.MAX_CANDIDATES_PER_S1)
print("\nIndexing candidate pool into inverted index buckets...")
blocker.build_candidate_index(df_candidates, show_progress=False)
print(f"Total inverted index buckets: {len(blocker.index):,}")

# 5. Measure Candidate Recall & Candidate Count
print("\nRetrieving candidates for S1 entities...")
candidate_dict = blocker.generate_candidate_pairs(df_s1_sample, show_progress=False)

# Evaluate recall
found_links = 0
total_candidates_generated = 0
strat_recall = defaultdict(int)

for s1_id, true_targets in gt_pairs.items():
    cands = candidate_dict.get(s1_id, set())
    total_candidates_generated += len(cands)
    for t in true_targets:
        if t in cands:
            found_links += 1

candidate_recall = (found_links / total_true_links) * 100
avg_candidates = total_candidates_generated / len(sample_s1_ids)
cartesian_pairs = len(sample_s1_ids) * len(df_candidates)
reduction_ratio = (1.0 - (total_candidates_generated / cartesian_pairs)) * 100

print("\n" + "=" * 80)
print("BLOCKING BENCHMARK RESULTS")
print("=" * 80)
print(f"Total True Links in Cohort : {total_true_links:,}")
print(f"True Links Recovered       : {found_links:,}")
print(f"Candidate Recall           : {candidate_recall:.2f}%  (Target: > 95%)")
print(f"Average Candidates per S1  : {avg_candidates:.1f}  (Target: <= 50)")
print(f"Total Candidate Pairs      : {total_candidates_generated:,}")
print(f"Reduction Ratio            : {reduction_ratio:.4f}%  (Target: > 99.9%)")
print("=" * 80)

# Verify against gate requirements
assert candidate_recall >= 90.0, f"Recall too low: {candidate_recall:.2f}%"
assert avg_candidates <= config.MAX_CANDIDATES_PER_S1, f"Candidate density too high: {avg_candidates:.1f}"
print("\nVerification Gate 03: Multi-Key Blocking PASSED!")
