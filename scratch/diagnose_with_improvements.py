import sys
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict
import re

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "code" / "business_entity_resolution" / "src"))

from blocking import generate_blocking_keys
from test_transliteration import transliterate_indic
import config

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

s1_records = {}
with open(config.TRAIN_SOURCE1, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in sample_s1_ids:
            s1_records[parts[0]] = parts
            if len(s1_records) == len(sample_s1_ids):
                break

target_records = {}
for src_path in [config.TRAIN_SOURCE2, config.TRAIN_SOURCE3]:
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in all_target_ids:
                target_records[parts[0]] = parts

def enhanced_keys(eid, name, addr, cntry):
    # Standard keys
    k = generate_blocking_keys(eid, name, addr, cntry)
    
    # 1. Spaceless key
    n_clean = re.sub(r'[^a-zA-Z0-9]', '', (name or '').lower())
    if len(n_clean) >= 4:
        k["B_SPACELESS"].append(f"{cntry}_spaceless_{n_clean}")
    
    # 2. Transliterated keys (if Indic text present)
    trans = transliterate_indic(name or "")
    if trans != name:
        k_trans = generate_blocking_keys(eid, trans, addr, cntry)
        for strat, key_list in k_trans.items():
            k[f"TRANS_{strat}"].extend(key_list)
            
    return k

shares = 0
missed = []
for s1_id, targets in gt_pairs.items():
    s1_row = s1_records.get(s1_id)
    if not s1_row:
        continue
    k1 = enhanced_keys(s1_id, s1_row[1], s1_row[2], s1_row[3])
    all_k1 = set().union(*k1.values())
    
    for t_id in targets:
        t_row = target_records.get(t_id)
        if not t_row:
            continue
        kt = enhanced_keys(t_id, t_row[1], t_row[2], t_row[3])
        all_kt = set().union(*kt.values())
        
        if all_k1 & all_kt:
            shares += 1
        else:
            missed.append((s1_row, t_row))

print(f"Original Union Recall: 98.86% (3,548 / 3,589)")
print(f"ENHANCED Union Recall: {shares / total_true_links * 100:.2f}% ({shares:,} / {total_true_links:,})")
print(f"Remaining Missed: {len(missed)}")
for idx, (s1, t) in enumerate(missed[:5], 1):
    print(f"\n--- Still Missed {idx} ---")
    print(f"S1: [{s1[0]}] ({s1[3]}) Name: {s1[1]} | Addr: {s1[2]}")
    print(f"T : [{t[0]}] ({t[3]}) Name: {t[1]} | Addr: {t[2]}")
