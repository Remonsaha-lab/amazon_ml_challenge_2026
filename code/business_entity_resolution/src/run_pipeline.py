"""
Master End-to-End Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
Trains the winning XGBoost matcher, streams test candidate generation, extracts
pairwise features, applies validation-optimal threshold tau=0.84, handles singletons,
and formats official matching_results.tsv and candidate_pairs.tsv submission files.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add package directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from normalization import (
    normalize_name,
    normalize_address,
    extract_postal_code,
    extract_numeric_tokens,
    normalize_country
)
from blocking import MultiKeyBlocker
from dataset import RecordCache, load_ground_truth_map, build_pairwise_dataset
from features import compute_pair_features
from models import train_xgboost
import config

print("=" * 80)
print("AMAZON ML CHALLENGE 2026: END-TO-END REPRODUCIBLE PIPELINE")
print("Business Entity Resolution — Winning XGBoost Architecture")
print("=" * 80)


def train_production_model(sample_s1_size: int = 15000):
    """
    Trains the production XGBoost classifier on representative training cohort
    with positive pairs and authentic mined hard negatives.
    """
    print(f"\n[Phase 1] Training Production Matcher on {sample_s1_size:,} S1 entities...")
    gt_map_full = load_ground_truth_map(config.TRAIN_GROUND_TRUTH)
    s1_cohort_ids = list(gt_map_full.keys())[:sample_s1_size]
    gt_map = {sid: gt_map_full[sid] for sid in s1_cohort_ids}
    cohort_set = set(s1_cohort_ids)

    # Load S1 records
    s1_rows = []
    with open(config.TRAIN_SOURCE1, "r", encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in cohort_set:
                s1_rows.append(parts)
                if len(s1_rows) == len(cohort_set):
                    break
    df_s1 = pd.DataFrame(s1_rows, columns=header)
    s1_cache = RecordCache(df_s1)

    # Load candidate pool
    all_target_ids = set()
    for sid in s1_cohort_ids:
        all_target_ids.update(gt_map[sid])

    cand_rows = []
    BACKGROUND_POOL = 150000
    c2 = c3 = 0
    with open(config.TRAIN_SOURCE2, "r", encoding="utf-8") as f:
        header_s2 = next(f).strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                if parts[0] in all_target_ids or c2 < (BACKGROUND_POOL // 2):
                    cand_rows.append(parts)
                    c2 += 1

    with open(config.TRAIN_SOURCE3, "r", encoding="utf-8") as f:
        header_s3 = next(f).strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                if parts[0] in all_target_ids or c3 < (BACKGROUND_POOL // 2):
                    cand_rows.append(parts)
                    c3 += 1

    df_cands = pd.DataFrame(cand_rows, columns=["entity_id", "business_name", "business_address", "country"])
    cand_cache = RecordCache(df_cands)

    # Index candidates
    blocker = MultiKeyBlocker(max_candidates_per_s1=config.MAX_CANDIDATES_PER_S1)
    blocker.build_candidate_index(df_cands, show_progress=False)
    cand_dict = blocker.generate_candidate_pairs(df_s1, show_progress=False)

    # Extract features with hard negatives
    X_train, y_train, _ = build_pairwise_dataset(
        s1_cache, cand_cache, cand_dict, gt_map, is_training=True, hard_neg_ratio=4
    )
    print(f"   Training Set Matrix: {X_train.shape} | Positives: {int(y_train.sum()):,}")

    model = train_xgboost(X_train, y_train)
    print("   Production XGBoost Model Trained Successfully!")
    return model


def run_inference_on_test(model, optimal_tau: float = 0.84, test_sample_limit: int = None):
    """
    Runs candidate generation and inference across test set, producing
    official matching_results.tsv and candidate_pairs.tsv.
    """
    print(f"\n[Phase 2] Running Inference on Test Dataset (Threshold tau={optimal_tau})...")
    
    # 1. Index Test Candidates (Source 2 and Source 3)
    print("   Indexing Test Candidate Pool from test_source2.tsv & test_source3.tsv...")
    test_cand_rows = []
    # Read test candidates
    for path in [config.TEST_SOURCE2, config.TEST_SOURCE3]:
        with open(path, "r", encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    test_cand_rows.append(parts)
                    
    df_test_cands = pd.DataFrame(test_cand_rows, columns=["entity_id", "business_name", "business_address", "country"])
    print(f"   Indexed {len(df_test_cands):,} candidate records.")
    
    test_cand_cache = RecordCache(df_test_cands)
    test_blocker = MultiKeyBlocker(max_candidates_per_s1=config.MAX_CANDIDATES_PER_S1)
    test_blocker.build_candidate_index(df_test_cands, show_progress=True)

    # 2. Process test_source1.tsv in streaming chunks
    print("   Processing test_source1.tsv entities...")
    matching_out = config.MATCHING_RESULTS_PATH
    candidate_out = config.CANDIDATE_PAIRS_PATH

    with open(matching_out, "w", encoding="utf-8") as f_match, \
         open(candidate_out, "w", encoding="utf-8") as f_cand:
        
        # Write exact required headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        with open(config.TEST_SOURCE1, "r", encoding="utf-8") as f_s1:
            header = next(f_s1).strip().split("\t")
            
            s1_count = 0
            for line in tqdm(f_s1, desc="Scoring Test S1 Entities"):
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                    
                s1_id = parts[0]
                s1_name = parts[1]
                s1_addr = parts[2]
                s1_country = parts[3]

                # Retrieve candidates via blocking
                cands = test_blocker.retrieve_candidates_for_record(
                    s1_id, s1_name, s1_addr, s1_country
                )
                
                # Write candidate list
                cand_list_str = ",".join(sorted(cands))
                f_cand.write(f"{s1_id}\t{cand_list_str}\n")

                if not cands:
                    # True singleton (no candidate survived)
                    f_match.write(f"{s1_id}\t\n")
                else:
                    # Extract pairwise features
                    s1_norm_name = normalize_name(s1_name)
                    s1_norm_addr = normalize_address(s1_addr)
                    s1_pin = extract_postal_code(s1_addr, s1_country)
                    s1_nums = extract_numeric_tokens(s1_addr)

                    cand_id_list = list(cands)
                    feat_matrix = []
                    for cid in cand_id_list:
                        vec = compute_pair_features(
                            s1_name=s1_name,
                            s1_addr=s1_addr,
                            s1_country=s1_country,
                            cand_id=cid,
                            cand_name=test_cand_cache.names[cid],
                            cand_addr=test_cand_cache.addrs[cid],
                            cand_country=test_cand_cache.countries[cid],
                            s1_norm_name=s1_norm_name,
                            s1_norm_addr=s1_norm_addr,
                            s1_pin=s1_pin,
                            s1_nums=s1_nums,
                            cand_norm_name=test_cand_cache.norm_names[cid],
                            cand_norm_addr=test_cand_cache.norm_addrs[cid],
                            cand_pin=test_cand_cache.pins[cid],
                            cand_nums=test_cand_cache.nums[cid]
                        )
                        feat_matrix.append(vec)

                    # Score with XGBoost
                    X_cand = np.array(feat_matrix, dtype=np.float32)
                    probs = model.predict_proba(X_cand)[:, 1]

                    # Decision layer: select candidates >= optimal_tau
                    matched = [cand_id_list[i] for i, p in enumerate(probs) if p >= optimal_tau]
                    
                    matched_str = ",".join(sorted(matched))
                    f_match.write(f"{s1_id}\t{matched_str}\n")

                s1_count += 1
                if test_sample_limit and s1_count >= test_sample_limit:
                    break

    print(f"\n[Phase 3] Output Generation Complete!")
    print(f"   Matching Results: {matching_out}")
    print(f"   Candidate Pairs : {candidate_out}")


if __name__ == "__main__":
    trained_model = train_production_model()
    # Test on initial cohort to verify validation script passes
    run_inference_on_test(trained_model, optimal_tau=0.84, test_sample_limit=2000)
