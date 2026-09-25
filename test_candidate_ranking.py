import sys
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import math

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "code" / "business_entity_resolution"))

from blocking import generate_blocking_keys
import config

print("=" * 80)
print("TESTING CANDIDATE RANKING & INVERSE BUCKET FREQUENCY RETRIEVAL")
print("=" * 80)

SAMPLE_SIZE = 1000
df_gt = pd.read_csv(config.TRAIN_GROUND_TRUTH, sep="\t", nrows=SAMPLE_SIZE * 5, dtype=str)
df_gt["matched_entity_ids"] = df_gt["matched_entity_ids"].fillna("")
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

# Load S1
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

# Load S2 and S3 candidate pool (including all targets + background pool of 100k)
cand_records = []
BACKGROUND_POOL = 100000
count_s2 = count_s3 = 0

with open(config.TRAIN_SOURCE2, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            if parts[0] in all_target_ids or count_s2 < (BACKGROUND_POOL // 2):
                cand_records.append(parts)
                count_s2 += 1

with open(config.TRAIN_SOURCE3, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            if parts[0] in all_target_ids or count_s3 < (BACKGROUND_POOL // 2):
                cand_records.append(parts)
                count_s3 += 1

df_candidates = pd.DataFrame(cand_records, columns=["entity_id", "business_name", "business_address", "country"])
print(f"Candidates in pool: {len(df_candidates):,}")

# Build Inverted Index with weights
index = defaultdict(set)
strategy_weights = {
    "B1": 15.0, # Exact Name
    "B2": 6.0,  # Name Token
    "B5": 5.0,  # Distinctive Number
    "B6": 4.0,  # Door + Locality
    "B4": 3.0,  # Postal Code / PIN
    "B3": 2.0,  # Name Prefix
    "B7": 1.0   # Distinctive Locality
}

for row in df_candidates.itertuples(index=False):
    c_keys = generate_blocking_keys(row.entity_id, str(row.business_name or ""), str(row.business_address or ""), str(row.country or ""))
    for strat, key_list in c_keys.items():
        for k in key_list:
            index[k].add(row.entity_id)

bucket_sizes = {k: len(v) for k, v in index.items()}
print(f"Total buckets: {len(index):,}")

# Test retrieval with top-N ranking
MAX_CANDS = 50
found_links = 0
total_candidates = 0

for row in df_s1_sample.itertuples(index=False):
    s1_id = row.entity_id
    s1_keys = generate_blocking_keys(s1_id, str(row.business_name or ""), str(row.business_address or ""), str(row.country or ""))
    
    cand_scores = defaultdict(float)
    
    for strat, key_list in s1_keys.items():
        weight = strategy_weights.get(strat, 1.0)
        for k in key_list:
            b_size = bucket_sizes.get(k, 0)
            if 0 < b_size <= 2500:
                # IDF-like attenuation: smaller buckets give much stronger signal
                idf_weight = weight / math.log2(2 + b_size)
                for cand_id in index[k]:
                    cand_scores[cand_id] += idf_weight
                    
    # Take top candidates by accumulated score
    top_cands = sorted(cand_scores.keys(), key=lambda x: cand_scores[x], reverse=True)[:MAX_CANDS]
    top_cands_set = set(top_cands)
    total_candidates += len(top_cands_set)
    
    true_targets = gt_pairs.get(s1_id, set())
    for t in true_targets:
        if t in top_cands_set:
            found_links += 1

recall = (found_links / total_true_links) * 100
avg_cands = total_candidates / len(sample_s1_ids)

print("\n" + "=" * 80)
print("RANKED CANDIDATE RETRIEVAL RESULTS")
print("=" * 80)
print(f"Candidate Recall         : {recall:.2f}% ({found_links:,} / {total_true_links:,})")
print(f"Average Candidates per S1: {avg_cands:.1f} (Cap: {MAX_CANDS})")
print(f"Total Candidate Pairs    : {total_candidates:,}")
print("=" * 80)
