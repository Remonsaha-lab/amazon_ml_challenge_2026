"""
Master End-to-End Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
Trains the winning XGBoost matcher with grouped train/val split, dynamically optimizes
decision threshold tau on the official Macro F0.5 metric, streams test candidate generation
with NaN protection, and formats official matching_results.tsv and candidate_pairs.tsv.
"""

import sys
import os
from pathlib import Path
import argparse
from typing import Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add package directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from normalization import (
    normalize_name,
    normalize_address,
    extract_postal_code,
    extract_numeric_tokens,
    normalize_country,
    is_empty_or_nan
)
from blocking import MultiKeyBlocker
from dataset import RecordCache, load_ground_truth_map, build_pairwise_dataset
from features import compute_pair_features
from models import train_xgboost
from decision import find_optimal_threshold, evaluate_macro_f05
import config

print("=" * 80)
print("AMAZON ML CHALLENGE 2026: END-TO-END REPRODUCIBLE PIPELINE")
print("Business Entity Resolution — Dynamic Threshold XGBoost Architecture")
print("=" * 80)


def train_production_model(sample_s1_size: int = 15000, max_cands: int = 50) -> Tuple[Any, float]:
    """
    Trains the production XGBoost classifier on an entity-grouped train/val split
    and dynamically grid-searches the threshold tau that maximizes Macro F0.5.
    Returns: (trained_model, optimal_tau)
    """
    print(f"\n[Phase 1] Training & Validating Production Matcher on {sample_s1_size:,} S1 entities...")
    gt_map_full = load_ground_truth_map(config.TRAIN_GROUND_TRUTH)
    s1_cohort_ids = list(gt_map_full.keys())[:sample_s1_size]
    gt_map = {sid: gt_map_full[sid] for sid in s1_cohort_ids}
    
    # 1. Entity-Level Grouped Split (80% train / 20% val)
    train_s1_ids, val_s1_ids = train_test_split(
        s1_cohort_ids, test_size=config.VALIDATION_SPLIT_RATIO, random_state=config.RANDOM_SEED
    )
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)
    print(f"   Train S1 Entities: {len(train_s1_set):,}")
    print(f"   Val S1 Entities  : {len(val_s1_set):,}")

    cohort_set = set(s1_cohort_ids)

    # 2. Load S1 records
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

    # 3. Load Candidate Records (Target matches + background candidate pool)
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

    # 4. Multi-Key Blocking
    blocker = MultiKeyBlocker(max_candidates_per_s1=max_cands)
    blocker.build_candidate_index(df_cands, show_progress=False)

    df_s1_train = df_s1[df_s1["entity_id"].isin(train_s1_set)]
    df_s1_val = df_s1[df_s1["entity_id"].isin(val_s1_set)]

    cand_train = blocker.generate_candidate_pairs(df_s1_train, show_progress=False)
    cand_val = blocker.generate_candidate_pairs(df_s1_val, show_progress=False)

    # 5. Extract Features (Training with hard negatives, Validation with all candidates)
    X_train, y_train, _ = build_pairwise_dataset(
        s1_cache, cand_cache, cand_train, gt_map, is_training=True, hard_neg_ratio=config.HARD_NEGATIVE_RATIO
    )
    X_val, y_val, val_pairs = build_pairwise_dataset(
        s1_cache, cand_cache, cand_val, gt_map, is_training=False
    )
    print(f"   X_train Matrix: {X_train.shape} | Positives: {int(y_train.sum()):,}")
    print(f"   X_val Matrix  : {X_val.shape} | Positives: {int(y_val.sum()):,}")

    # 6. Train Model
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # 7. Optimize Threshold tau directly on Macro F0.5
    val_probs = model.predict_proba(X_val)[:, 1]
    val_pair_scores = [
        (val_pairs[i][0], val_pairs[i][1], float(val_probs[i]))
        for i in range(len(val_pairs))
    ]
    optimal_tau, val_f05, _ = find_optimal_threshold(gt_map, val_pair_scores, val_s1_set)
    print(f"   Dynamic Threshold Search Complete -> Optimal tau = {optimal_tau:.3f} (Validation Macro F0.5: {val_f05:.4f})")
    
    return model, optimal_tau


def run_inference_on_test(model, optimal_tau: float, test_sample_limit: Optional[int] = None, max_cands: int = 50):
    """
    Runs candidate generation and inference across test set, producing
    official matching_results.tsv and candidate_pairs.tsv.
    """
    scope_desc = f"first {test_sample_limit:,}" if test_sample_limit else "FULL 1.73M test set"
    print(f"\n[Phase 2] Running Inference on Test Dataset ({scope_desc}, Threshold tau={optimal_tau:.3f})...")
    
    # 1. Index Test Candidates (Source 2 and Source 3)
    print("   Indexing Test Candidate Pool from test_source2.tsv & test_source3.tsv...")
    test_cand_rows = []
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
    test_blocker = MultiKeyBlocker(max_candidates_per_s1=max_cands)
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
                    
                s1_id = parts[0].strip()
                s1_name = "" if is_empty_or_nan(parts[1]) else parts[1].strip()
                s1_addr = "" if is_empty_or_nan(parts[2]) else parts[2].strip()
                s1_country = "" if is_empty_or_nan(parts[3]) else parts[3].strip()

                # Retrieve candidates via blocking (ranked list)
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
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: Business Entity Resolution Master Pipeline")
    parser.add_argument("--full", action="store_true", help="Run full test inference across all 1.73M test records")
    parser.add_argument("--test-sample", type=int, default=2000, help="Number of test records to process (default: 2000 for quick verification, use --full for all)")
    parser.add_argument("--train-cohort", type=int, default=15000, help="Number of S1 entities to train/validate on (default: 15000)")
    parser.add_argument("--max-cands", type=int, default=50, help="Max candidates per S1 entity in blocking (default: 50)")
    args = parser.parse_args()

    test_limit = None if args.full else args.test_sample

    trained_model, optimal_tau = train_production_model(sample_s1_size=args.train_cohort, max_cands=args.max_cands)
    run_inference_on_test(trained_model, optimal_tau=optimal_tau, test_sample_limit=test_limit, max_cands=args.max_cands)
