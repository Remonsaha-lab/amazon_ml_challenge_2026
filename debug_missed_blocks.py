import sys
import os
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "code" / "business_entity_resolution" / "src"))


from blocking import MultiKeyBlocker, generate_blocking_keys
import config

# Load small sample of GT
# Sample 1,000 entities to capture authentic missed links
df_gt = pd.read_csv(config.TRAIN_GROUND_TRUTH, sep="\t", nrows=5000, dtype=str)
df_gt["matched_entity_ids"] = df_gt["matched_entity_ids"].fillna("")
df_gt = df_gt[df_gt["matched_entity_ids"].str.strip() != ""].head(1000)


sample_s1_ids = set(df_gt["source1_entity_id"])
all_target_ids = set()
gt_pairs = {}
for _, row in df_gt.iterrows():
    s1_id = row["source1_entity_id"]
    targets = {x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()}
    gt_pairs[s1_id] = targets
    all_target_ids.update(targets)

# Load S1
s1_records = {}
with open(config.TRAIN_SOURCE1, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in sample_s1_ids:
            s1_records[parts[0]] = parts

# Load targets from S2 and S3
cand_records = {}
with open(config.TRAIN_SOURCE2, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in all_target_ids:
            cand_records[parts[0]] = parts
            if len(cand_records) == len(all_target_ids):
                break

with open(config.TRAIN_SOURCE3, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in all_target_ids:
            cand_records[parts[0]] = parts
            if len(cand_records) == len(all_target_ids):
                break

# Build index of these targets
df_cand = pd.DataFrame(list(cand_records.values()), columns=["entity_id", "business_name", "business_address", "country"])
blocker = MultiKeyBlocker(max_candidates_per_s1=100)
blocker.build_candidate_index(df_cand, show_progress=False)

print("\n" + "=" * 90)
print("ANALYSIS OF MISSED TRUE LINKS IN BLOCKING")
print("=" * 90)

missed_count = 0
for s1_id, true_targets in gt_pairs.items():
    s1_row = s1_records.get(s1_id)
    if not s1_row:
        continue
    cands = blocker.retrieve_candidates_for_record(s1_id, s1_row[1], s1_row[2], s1_row[3])
    s1_keys = generate_blocking_keys(s1_id, s1_row[1], s1_row[2], s1_row[3])
    
    for t_id in true_targets:
        if t_id not in cands and t_id in cand_records:
            missed_count += 1
            t_row = cand_records[t_id]
            t_keys = generate_blocking_keys(t_id, t_row[1], t_row[2], t_row[3])
            
            # Find key overlap
            all_s1_k = {k for kl in s1_keys.values() for k in kl}
            all_t_k = {k for kl in t_keys.values() for k in kl}
            common = all_s1_k & all_t_k
            
            print(f"\n[Missed Pair {missed_count}] S1: {s1_id} <---> Target: {t_id}")
            print(f"  S1 Name       : {s1_row[1]}")
            print(f"  Target Name   : {t_row[1]}")
            print(f"  S1 Address    : {s1_row[2]}")
            print(f"  Target Address: {t_row[2]}")
            print(f"  S1 Keys       : {dict(s1_keys)}")
            print(f"  Target Keys   : {dict(t_keys)}")
            print(f"  Common Keys   : {common}")
            if missed_count == 0:
                print("\nNo missed links found in this sample — 100% recall on the evaluated entities!")
            else:
                print(f"\nTotal missed pairs displayed: {missed_count}")

            
            if missed_count >= 5:
                break
    if missed_count >= 5:
        break
