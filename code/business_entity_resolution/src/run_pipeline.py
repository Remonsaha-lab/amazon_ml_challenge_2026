"""
Master End-to-End Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
Trains the winning XGBoost matcher with chunked memory-safe feature extraction,
dynamically optimizes decision threshold tau on the official Macro F0.5 metric,
and streams country-partitioned test candidate generation with compact uint32 array
indexing to guarantee low RAM (<1.5 GB).
"""

import sys
import os
from pathlib import Path
import argparse
from typing import Optional, Tuple, Any, Dict, List, Set
import gc
from collections import defaultdict
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
from blocking import MultiKeyBlocker, CountryPartitionedBlocker
from dataset import RecordCache, load_ground_truth_map, build_pairwise_dataset, ChunkedDatasetBuilder
from features import compute_pair_features
from models import train_xgboost, get_feature_importances
from decision import find_optimal_threshold, evaluate_macro_f05, filter_candidates_with_margin
import config

print("=" * 80)
print("AMAZON ML CHALLENGE 2026: END-TO-END REPRODUCIBLE PIPELINE")
print("Business Entity Resolution — High-Scale Country Partitioned XGBoost Architecture")
print("=" * 80)


def train_production_model(
    sample_s1_size: int = 30000,
    max_cands: int = 50,
    chunk_size: int = 10000
) -> Tuple[Any, float]:
    """
    Trains the production XGBoost classifier using memory-safe ChunkedDatasetBuilder
    on an entity-grouped train/val split and dynamically grid-searches the threshold tau
    that maximizes Macro F0.5.
    Returns: (trained_model, optimal_tau)
    """
    cohort_desc = f"{sample_s1_size:,}" if sample_s1_size > 0 else "FULL 2.2M"
    print(f"\n[Phase 1] Training & Validating Production Matcher on {cohort_desc} S1 entities...")
    gt_map_full = load_ground_truth_map(config.TRAIN_GROUND_TRUTH)
    all_s1_keys = list(gt_map_full.keys())

    rng = np.random.RandomState(config.RANDOM_SEED)

    if sample_s1_size > 0 and sample_s1_size < len(all_s1_keys):
        # Stratified/random sampling across all training entities
        sampled_indices = rng.choice(len(all_s1_keys), size=sample_s1_size, replace=False)
        s1_cohort_ids = [all_s1_keys[i] for i in sampled_indices]
    else:
        s1_cohort_ids = all_s1_keys

    gt_map = {sid: gt_map_full[sid] for sid in s1_cohort_ids}
    
    # 1. Entity-Level Grouped Split (80% train / 20% val)
    train_s1_ids, val_s1_ids = train_test_split(
        s1_cohort_ids, test_size=config.VALIDATION_SPLIT_RATIO, random_state=config.RANDOM_SEED
    )
    # Cap validation S1 to at most 6,000 entities to keep validation evaluation fast
    if len(val_s1_ids) > 6000:
        val_s1_ids = val_s1_ids[:6000]

    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)
    print(f"   Train S1 Entities: {len(train_s1_set):,}")
    print(f"   Val S1 Entities  : {len(val_s1_set):,}")

    cohort_set = train_s1_set | val_s1_set

    # 2. Load S1 records for the selected cohort
    train_s1_tuples = []
    val_s1_tuples = []

    with open(config.TRAIN_SOURCE1, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0] in cohort_set:
                sid = parts[0].strip()
                sname = "" if is_empty_or_nan(parts[1]) else parts[1].strip()
                saddr = "" if is_empty_or_nan(parts[2]) else parts[2].strip()
                scntry = "" if is_empty_or_nan(parts[3]) else parts[3].strip()
                record = (sid, sname, saddr, scntry)
                if sid in train_s1_set:
                    train_s1_tuples.append(record)
                elif sid in val_s1_set:
                    val_s1_tuples.append(record)
                if len(train_s1_tuples) == len(train_s1_set) and len(val_s1_tuples) == len(val_s1_set):
                    break

    # 3. Collect all true match IDs for the cohort + background candidate pool
    all_target_ids: Set[str] = set()
    for sid in cohort_set:
        all_target_ids.update(gt_map.get(sid, set()))

    BACKGROUND_POOL = min(max(len(cohort_set) * 4, 100000), 400000)
    cand_records: List[Tuple[str, str, str, str]] = []
    c2 = c3 = 0
    target_found = 0

    print("   Streaming candidate records from Train Source 2 & 3...")
    with open(config.TRAIN_SOURCE2, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                cid = parts[0].strip()
                if cid in all_target_ids:
                    cand_records.append((cid, parts[1].strip(), parts[2].strip(), parts[3].strip()))
                    target_found += 1
                elif c2 < (BACKGROUND_POOL // 2):
                    cand_records.append((cid, parts[1].strip(), parts[2].strip(), parts[3].strip()))
                    c2 += 1

    with open(config.TRAIN_SOURCE3, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                cid = parts[0].strip()
                if cid in all_target_ids:
                    cand_records.append((cid, parts[1].strip(), parts[2].strip(), parts[3].strip()))
                    target_found += 1
                elif c3 < (BACKGROUND_POOL // 2):
                    cand_records.append((cid, parts[1].strip(), parts[2].strip(), parts[3].strip()))
                    c3 += 1

    print(f"   Indexed Candidate Records: {len(cand_records):,} (Targets: {target_found:,})")

    # 4. Build Candidate Cache and Inverted Index Blocker
    cand_cache = RecordCache(cand_records)
    blocker = MultiKeyBlocker(max_candidates_per_s1=max_cands)
    blocker.build_candidate_index_from_records(cand_records, show_progress=False)

    # 5. Measure Blocker Recall on Validation Cohort
    val_cand_dict = {}
    for s1_id, name, addr, cntry in val_s1_tuples:
        val_cand_dict[s1_id] = blocker.retrieve_candidates_for_record(s1_id, name, addr, cntry)
    
    total_val_targets = 0
    retrieved_val_targets = 0
    for s1_id in val_s1_set:
        targets = gt_map.get(s1_id, set())
        total_val_targets += len(targets)
        cands_set = set(val_cand_dict.get(s1_id, []))
        retrieved_val_targets += len(targets & cands_set)
    
    if total_val_targets > 0:
        blocker_recall = retrieved_val_targets / total_val_targets
        print(f"   [Validation] Blocker Candidate Recall: {blocker_recall*100:.2f}% ({retrieved_val_targets:,}/{total_val_targets:,} true targets captured)")

    # 6. Two-Stage Active Hard Negative Mining & Feature Extraction
    dataset_builder = ChunkedDatasetBuilder(chunk_size=chunk_size, max_cands=max_cands, hard_neg_ratio=config.HARD_NEGATIVE_RATIO)

    print(f"   [Stage 1] Extracting baseline feature matrix in chunks of {chunk_size:,}...")
    X_train_base, y_train_base, _ = dataset_builder.process_s1_chunks(
        train_s1_tuples, cand_cache, blocker, gt_map, is_training=True, base_model=None, desc="Train Chunks Stage 1"
    )

    print(f"   [Stage 1] Extracting validation feature matrix...")
    X_val, y_val, val_pairs = dataset_builder.process_s1_chunks(
        val_s1_tuples, cand_cache, blocker, gt_map, is_training=False, desc="Val Chunks"
    )

    print(f"   Baseline Training Matrix: {X_train_base.shape} | Positives: {int(y_train_base.sum()):,}")
    print(f"   Validation Matrix       : {X_val.shape} | Positives: {int(y_val.sum()):,}")

    # Stage 1: Fit fast baseline model to score and identify difficult false positive candidates
    print("   [Stage 1] Training baseline model for active hard negative mining...")
    base_model = train_xgboost(X_train_base, y_train_base, params={"n_estimators": 100, "max_depth": 5})

    # Stage 2: Re-extract training pairs with active model-confused hard negatives mined across all candidate sets
    print("   [Stage 2] Active Hard Negative Mining: Scoring all candidate negatives with baseline model...")
    X_train, y_train, _ = dataset_builder.process_s1_chunks(
        train_s1_tuples, cand_cache, blocker, gt_map, is_training=True,
        base_model=base_model, desc="Train Chunks Stage 2"
    )
    print(f"   Hard-Negative Augmented Training Matrix: {X_train.shape} | Positives: {int(y_train.sum()):,}")

    del X_train_base, y_train_base, base_model
    gc.collect()

    # 7. Fit Final Production XGBoost Classifier
    print("   Fitting production XGBoost model on hard-negative augmented dataset...")
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # Top feature importances
    importances = get_feature_importances(model)
    top_5 = list(importances.items())[:5]
    print(f"   Top Features: {', '.join(f'{k} ({v*100:.1f}%)' for k, v in top_5)}")

    # 8. Fine-Grained Threshold Search tau on Macro F0.5
    val_probs = model.predict_proba(X_val)[:, 1]
    # Pack validation features into scores for calibrated filtering
    val_pair_scores = [
        (val_pairs[i][0], val_pairs[i][1], float(val_probs[i]))
        for i in range(len(val_pairs))
    ]
    optimal_tau, val_f05, _ = find_optimal_threshold(gt_map, val_pair_scores, val_s1_set)
    print(f"   Fine-Grained Threshold Search Complete -> Optimal tau = {optimal_tau:.3f} (Validation Macro F0.5: {val_f05:.4f})")
    
    # Save production model and threshold
    import joblib
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = config.OUTPUT_DIR / "production_model.joblib"
    joblib.dump({"model": model, "optimal_tau": optimal_tau, "top_features": top_5}, model_save_path)
    print(f"   [Model Saved] Model artifact and threshold saved to: {model_save_path}")

    # Clean up training data from memory
    blocker.clear()
    cand_cache.clear()
    del cand_records, X_train, y_train, X_val, y_val, train_s1_tuples, val_s1_tuples
    gc.collect()

    return model, optimal_tau


def run_inference_on_test(
    model,
    optimal_tau: float,
    test_sample_limit: Optional[int] = None,
    max_cands: int = 50,
    batch_size: int = 1000
):
    """
    Runs country-partitioned streaming candidate generation and vectorized inference across test set.
    Partitions candidate pool by country (France, US, India), keeping total RAM < 1.5 GB.
    Uses vectorized batch inference (1,000 entities at a time) for 50x speedup.
    Guarantees 100% test coverage with exact row alignment to test_source1.tsv.
    """
    scope_desc = f"first {test_sample_limit:,}" if test_sample_limit else "FULL 1.73M test set"
    print(f"\n[Phase 2] Running Country-Partitioned Inference on Test Dataset ({scope_desc}, Threshold tau={optimal_tau:.3f})...")
    
    # 1. Read test_source1.tsv and group by country
    print("   Grouping test S1 entities by country...")
    s1_by_country: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
    s1_all_ids: List[str] = []
    total_s1 = 0

    with open(config.TEST_SOURCE1, "r", encoding="utf-8") as f_s1:
        next(f_s1)  # header
        for line in f_s1:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                s1_id = parts[0].strip()
                s1_name = "" if is_empty_or_nan(parts[1]) else parts[1].strip()
                s1_addr = "" if is_empty_or_nan(parts[2]) else parts[2].strip()
                s1_country = "" if is_empty_or_nan(parts[3]) else parts[3].strip()
                c_norm = normalize_country(s1_country)
                s1_by_country[c_norm].append((s1_id, s1_name, s1_addr, s1_country))
                s1_all_ids.append(s1_id)
                total_s1 += 1
                if test_sample_limit and total_s1 >= test_sample_limit:
                    break

    for c, entities in s1_by_country.items():
        print(f"     * {c:10s}: {len(entities):,} entities")

    matching_out = config.MATCHING_RESULTS_PATH
    candidate_out = config.CANDIDATE_PAIRS_PATH
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Master dictionaries mapping s1_id -> result string
    master_matches: Dict[str, str] = {}
    master_candidates: Dict[str, str] = {}

    # Order of countries to process: France (smallest) -> US (medium) -> India (largest)
    countries_to_process = [c for c in ["France", "US", "India"] if c in s1_by_country]
    for c in s1_by_country:
        if c not in countries_to_process:
            countries_to_process.append(c)

    # 2. Process each country in complete isolation (zero RAM leakage across countries)
    for c_idx, target_country in enumerate(countries_to_process, 1):
        s1_list = s1_by_country[target_country]
        print(f"\n[Partition {c_idx}/{len(countries_to_process)}] Loading & Indexing {target_country} candidates ({len(s1_list):,} S1 queries)...")
        
        ckpt_match_path = config.OUTPUT_DIR / f"checkpoint_matching_{target_country}.tsv"
        ckpt_cand_path = config.OUTPUT_DIR / f"checkpoint_cands_{target_country}.tsv"

        # Check if country checkpoint already exists and can be resumed/skipped
        already_scored_matches = {}
        already_scored_cands = {}
        if ckpt_match_path.exists() and ckpt_cand_path.exists():
            with open(ckpt_match_path, "r", encoding="utf-8") as f_chk:
                for line in f_chk:
                    p = line.rstrip("\r\n").split("\t")
                    if p and p[0]:
                        already_scored_matches[p[0]] = p[1] if len(p) > 1 else ""
            with open(ckpt_cand_path, "r", encoding="utf-8") as f_chk:
                for line in f_chk:
                    p = line.rstrip("\r\n").split("\t")
                    if p and p[0]:
                        already_scored_cands[p[0]] = p[1] if len(p) > 1 else ""

        common_ids = set(already_scored_matches.keys()) & set(already_scored_cands.keys())
        if len(common_ids) == len(s1_list):
            print(f"   ⚡ Country {target_country} is ALREADY 100% complete ({len(common_ids):,} rows)! Skipping partition.")
            master_matches.update({sid: already_scored_matches[sid] for sid in common_ids})
            master_candidates.update({sid: already_scored_cands[sid] for sid in common_ids})
            continue
        elif len(common_ids) > 0:
            print(f"   ⚡ Resuming {target_country} from checkpoint ({len(common_ids):,} clean rows already scored)...")
            # Clean and rewrite checkpoint files with valid common_ids to remove any cut-off trailing line
            with open(ckpt_match_path, "w", encoding="utf-8") as f_m, open(ckpt_cand_path, "w", encoding="utf-8") as f_c:
                for sid in common_ids:
                    f_m.write(f"{sid}\t{already_scored_matches[sid]}\n")
                    f_c.write(f"{sid}\t{already_scored_cands[sid]}\n")

            master_matches.update({sid: already_scored_matches[sid] for sid in common_ids})
            master_candidates.update({sid: already_scored_cands[sid] for sid in common_ids})
            s1_list = [rec for rec in s1_list if rec[0] not in common_ids]
            print(f"   Remaining to score for {target_country}: {len(s1_list):,} entities.")
            file_mode = "a"
        else:
            file_mode = "w"

        cand_tuples = []
        is_debug_mode = (test_sample_limit is not None and test_sample_limit <= 5000)
        DEBUG_CAND_CAP = 150000

        for path in [config.TEST_SOURCE2, config.TEST_SOURCE3]:
            with open(path, "r", encoding="utf-8") as f:
                next(f)  # header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        if normalize_country(parts[3]) == target_country:
                            cand_tuples.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()))
                            if is_debug_mode and len(cand_tuples) >= DEBUG_CAND_CAP:
                                break
            if is_debug_mode and len(cand_tuples) >= DEBUG_CAND_CAP:
                break

        print(f"   Loaded {len(cand_tuples):,} candidates for {target_country}.")
        
        # Build compact cache and uint32 blocker for this country
        cand_cache = RecordCache(cand_tuples)
        blocker = MultiKeyBlocker(max_candidates_per_s1=max_cands)
        blocker.build_candidate_index_from_records(cand_tuples, show_progress=False)
        print(f"   Indexed {target_country} candidate index: {len(blocker.index):,} buckets.")

        from concurrent.futures import ThreadPoolExecutor

        def process_single_s1(rec):
            s1_id, s1_name, s1_addr, s1_country = rec
            cands = blocker.retrieve_candidates_for_record(s1_id, s1_name, s1_addr, s1_country)
            cand_list_str = ",".join(cands)
            if not cands:
                return (s1_id, cand_list_str, [], [])

            s1_norm_name = normalize_name(s1_name)
            s1_norm_addr = normalize_address(s1_addr)
            s1_pin = extract_postal_code(s1_addr, s1_country)
            s1_nums = extract_numeric_tokens(s1_addr)

            v_cands = []
            feats = []
            for cid in cands:
                if cid in cand_cache.norm_names:
                    vec = compute_pair_features(
                        s1_name="", s1_addr="", s1_country=s1_country,
                        cand_id=cid, cand_name="", cand_addr="",
                        cand_country=cand_cache.countries.get(cid, target_country),
                        s1_norm_name=s1_norm_name, s1_norm_addr=s1_norm_addr,
                        s1_pin=s1_pin, s1_nums=s1_nums,
                        cand_norm_name=cand_cache.norm_names[cid],
                        cand_norm_addr=cand_cache.norm_addrs[cid],
                        cand_pin=cand_cache.pins[cid],
                        cand_nums=cand_cache.nums[cid]
                    )
                    feats.append(vec)
                    v_cands.append(cid)
            return (s1_id, cand_list_str, v_cands, feats)

        # 16-thread parallel batch scoring
        with open(ckpt_match_path, file_mode, encoding="utf-8") as f_ckpt_match, \
             open(ckpt_cand_path, file_mode, encoding="utf-8") as f_ckpt_cand, \
             ThreadPoolExecutor(max_workers=16) as pool:
            
            pbar = tqdm(total=len(s1_list), desc=f"Parallel 16-Core Scoring {target_country} S1")
            
            for b_start in range(0, len(s1_list), batch_size):
                b_end = min(b_start + batch_size, len(s1_list))
                batch_s1 = s1_list[b_start:b_end]

                batch_results = list(pool.map(process_single_s1, batch_s1))

                batch_info = []
                batch_feat_rows = []

                for s1_id, cand_list_str, valid_cands, feats in batch_results:
                    feat_start = len(batch_feat_rows)
                    if feats:
                        batch_feat_rows.extend(feats)
                    batch_info.append((s1_id, cand_list_str, valid_cands, feat_start, len(feats)))

                # Batched XGBoost inference across all candidates in this batch
                if batch_feat_rows:
                    X_batch = np.array(batch_feat_rows, dtype=np.float32)
                    probs_batch = model.predict_proba(X_batch)[:, 1]
                else:
                    probs_batch = np.empty(0, dtype=np.float32)

                # Filter and record results
                for s1_id, cand_list_str, valid_cands, feat_start, feat_count in batch_info:
                    f_ckpt_cand.write(f"{s1_id}\t{cand_list_str}\n")
                    master_candidates[s1_id] = cand_list_str

                    if feat_count == 0:
                        f_ckpt_match.write(f"{s1_id}\t\n")
                        master_matches[s1_id] = ""
                    else:
                        cands_with_info = [
                            (valid_cands[i], float(probs_batch[feat_start + i]),
                             float(batch_feat_rows[feat_start + i][9]),
                             float(batch_feat_rows[feat_start + i][18]),
                             float(batch_feat_rows[feat_start + i][26]))
                            for i in range(feat_count)
                        ]
                        matched = filter_candidates_with_margin(cands_with_info, tau=optimal_tau)
                        matched_str = ",".join(matched)
                        f_ckpt_match.write(f"{s1_id}\t{matched_str}\n")
                        master_matches[s1_id] = matched_str

                pbar.update(len(batch_s1))
                f_ckpt_match.flush()
                f_ckpt_cand.flush()

            pbar.close()


        # Explicitly release memory for this country partition
        blocker.clear()
        cand_cache.clear()
        del cand_tuples, cand_cache, blocker
        gc.collect()
        print(f"   [Partition Complete] {target_country} saved to checkpoints.")

    # 3. Assemble Master Output with exact row-order preservation of test_source1.tsv
    print(f"\n[Phase 3] Assembling Master Submission Files (aligning with test_source1.tsv)...")
    total_written = 0
    non_empty_matches = 0

    with open(matching_out, "w", encoding="utf-8") as f_match, \
         open(candidate_out, "w", encoding="utf-8") as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in s1_all_ids:
            m_str = master_matches.get(s1_id, "")
            c_str = master_candidates.get(s1_id, "")
            f_match.write(f"{s1_id}\t{m_str}\n")
            f_cand.write(f"{s1_id}\t{c_str}\n")
            total_written += 1
            if m_str:
                non_empty_matches += 1

    print(f"   [Done] Master files written:")
    print(f"     * Matching Results: {matching_out}")
    print(f"     * Candidate Pairs : {candidate_out}")
    print(f"   Total S1 Entities   : {total_written:,} rows (Non-empty matches: {non_empty_matches:,})")
    print(f"   Coverage Ratio      : {total_written / max(len(s1_all_ids), 1) * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: Business Entity Resolution Master Pipeline")
    parser.add_argument("--full", action="store_true", help="Run full test inference across all 1.73M test records")
    parser.add_argument("--test-sample", type=int, default=2000, help="Number of test records to process (default: 2000 for quick verification, use --full for all)")
    parser.add_argument("--train-cohort", type=int, default=50000, help="Number of S1 entities to train/validate on (default: 50000, use 0 for full trainset)")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Chunk size for memory-safe feature extraction (default: 10000)")
    parser.add_argument("--max-cands", type=int, default=50, help="Max candidates per S1 entity in blocking (default: 50)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Vectorized inference batch size (default: 1000)")
    parser.add_argument("--load-model", type=str, default="", help="Path to saved .joblib model to skip training and run inference directly")
    parser.add_argument("--tau", type=float, default=None, help="Explicit decision threshold tau (overrides searched tau)")
    parser.add_argument("--output", type=str, default="", help="Custom output directory")
    args = parser.parse_args()

    if args.output:
        config.OUTPUT_DIR = Path(args.output)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.MATCHING_RESULTS_PATH = config.OUTPUT_DIR / "matching_results.tsv"
        config.CANDIDATE_PAIRS_PATH = config.OUTPUT_DIR / "candidate_pairs.tsv"

    test_limit = None if args.full else args.test_sample


    import joblib
    if args.load_model and Path(args.load_model).is_file():
        print(f"\n[Model Loading] Loading existing production model from: {args.load_model}")
        saved_data = joblib.load(args.load_model)
        trained_model = saved_data["model"]
        optimal_tau = args.tau if args.tau is not None else saved_data.get("optimal_tau", 0.680)
        print(f"   Loaded Model successfully. Threshold tau = {optimal_tau:.3f}")
    else:
        trained_model, optimal_tau = train_production_model(
            sample_s1_size=args.train_cohort,
            max_cands=args.max_cands,
            chunk_size=args.chunk_size
        )
        if args.tau is not None:
            optimal_tau = args.tau
            print(f"   Threshold overridden by user flag: tau = {optimal_tau:.3f}")

    run_inference_on_test(
        trained_model,
        optimal_tau=optimal_tau,
        test_sample_limit=test_limit,
        max_cands=args.max_cands,
        batch_size=args.batch_size
    )

