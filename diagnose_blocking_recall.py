import sys
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "code" / "business_entity_resolution" / "src"))

from blocking import generate_blocking_keys
import config

print("=" * 80)
print("DIAGNOSING THEORETICAL BLOCKING RECALL & BOTTLENECK ANALYSIS")
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
print(f"Sample S1 count: {len(sample_s1_ids)}")
print(f"Total True Links: {total_true_links}")

# Load S1 records
s1_records = {}
with open(config.TRAIN_SOURCE1, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in sample_s1_ids:
            s1_records[parts[0]] = parts
            if len(s1_records) == len(sample_s1_ids):
                break

# Load target records
target_records = {}
for src_path in [config.TRAIN_SOURCE2, config.TRAIN_SOURCE3]:
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in all_target_ids:
                target_records[parts[0]] = parts

print(f"Loaded {len(s1_records)} S1 and {len(target_records)} Target records.")

# Evaluate key sharing for each true link
shares_b1 = 0
shares_b2 = 0
shares_b3 = 0
shares_b4 = 0
shares_b5 = 0
shares_b6 = 0
shares_b7 = 0
shares_any = 0

missed_any = []

for s1_id, targets in gt_pairs.items():
    s1_row = s1_records.get(s1_id)
    if not s1_row:
        continue
    k1 = generate_blocking_keys(s1_id, s1_row[1], s1_row[2], s1_row[3])
    
    for t_id in targets:
        t_row = target_records.get(t_id)
        if not t_row:
            continue
        kt = generate_blocking_keys(t_id, t_row[1], t_row[2], t_row[3])
        
        has_b1 = bool(set(k1.get("B1", [])) & set(kt.get("B1", [])))
        has_b2 = bool(set(k1.get("B2", [])) & set(kt.get("B2", [])))
        has_b3 = bool(set(k1.get("B3", [])) & set(kt.get("B3", [])))
        has_b4 = bool(set(k1.get("B4", [])) & set(kt.get("B4", [])))
        has_b5 = bool(set(k1.get("B5", [])) & set(kt.get("B5", [])))
        has_b6 = bool(set(k1.get("B6", [])) & set(kt.get("B6", [])))
        has_b7 = bool(set(k1.get("B7", [])) & set(kt.get("B7", [])))
        
        if has_b1: shares_b1 += 1
        if has_b2: shares_b2 += 1
        if has_b3: shares_b3 += 1
        if has_b4: shares_b4 += 1
        if has_b5: shares_b5 += 1
        if has_b6: shares_b6 += 1
        if has_b7: shares_b7 += 1
        
        has_any = has_b1 or has_b2 or has_b3 or has_b4 or has_b5 or has_b6 or has_b7
        if has_any:
            shares_any += 1
        else:
            missed_any.append((s1_row, t_row))

print("\n" + "=" * 80)
print("THEORETICAL RECALL CEILING PER KEY TYPE (ZERO BUCKET LIMIT)")
print("=" * 80)
print(f"B1 (Exact Name)          : {shares_b1 / total_true_links * 100:.2f}% ({shares_b1:,} links)")
print(f"B2 (Any Name Token)      : {shares_b2 / total_true_links * 100:.2f}% ({shares_b2:,} links)")
print(f"B3 (Name 4-Prefix)       : {shares_b3 / total_true_links * 100:.2f}% ({shares_b3:,} links)")
print(f"B4 (Postal Code / PIN)   : {shares_b4 / total_true_links * 100:.2f}% ({shares_b4:,} links)")
print(f"B5 (Numbers >= 4 digits) : {shares_b5 / total_true_links * 100:.2f}% ({shares_b5:,} links)")
print(f"B6 (Door + Locality)     : {shares_b6 / total_true_links * 100:.2f}% ({shares_b6:,} links)")
print(f"B7 (Locality Token >= 6) : {shares_b7 / total_true_links * 100:.2f}% ({shares_b7:,} links)")
print("-" * 80)
print(f"THEORETICAL UNION RECALL : {shares_any / total_true_links * 100:.2f}% ({shares_any:,} / {total_true_links:,})")
print("=" * 80)

print(f"\nSample of {len(missed_any)} true pairs that shared ZERO keys across B1..B7:")
for idx, (s1, t) in enumerate(missed_any[:5], 1):
    print(f"\n--- Missed Pair {idx} ---")
    print(f"S1: [{s1[0]}] ({s1[3]}) Name: {s1[1]} | Addr: {s1[2]}")
    print(f"T : [{t[0]}] ({t[3]}) Name: {t[1]} | Addr: {t[2]}")
