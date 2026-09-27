import sys
from pathlib import Path
import random
import numpy as np
from collections import defaultdict
import joblib

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "code_1"))
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))

import config
from dataset import load_ground_truth_map, RecordCache
from blocking import MultiKeyBlocker
from decision import evaluate_macro_f05
from features import compute_pair_features, FEATURE_NAMES
from normalization import (
    extract_numeric_tokens,
    extract_postal_code,
    normalize_address,
    normalize_name,
)
from rank_pipeline import read_source, get_s1_records, collect_training_candidates, make_rank_matrix

def run():
    print("Loading ground truth...")
    full_gt = load_ground_truth_map(config.TRAIN_GROUND_TRUTH)
    all_ids = list(full_gt)
    rng = random.Random(42)
    cohort_ids = rng.sample(all_ids, 5000)
    
    # 80/20 split
    train_ids = set(cohort_ids[:4000])
    val_ids = set(cohort_ids[4000:])
    cohort = train_ids | val_ids
    gt = {sid: full_gt[sid] for sid in cohort}
    
    val_s1 = get_s1_records(config.TRAIN_SOURCE1, val_ids)
    bg = 50_000
    candidate_records = collect_training_candidates(cohort, gt, bg)
    cache = RecordCache(candidate_records)
    blocker = MultiKeyBlocker(max_candidates_per_s1=100)
    blocker.build_candidate_index_from_records(candidate_records, show_progress=False)
    
    print(f"Building val rank matrix for {len(val_s1)} entities...")
    x_val, y_val, group_val, val_pair_groups = make_rank_matrix(
        val_s1, cache, blocker, gt, 100, keep_pairs=True
    )
    
    # Check candidate blocker recall directly on validation
    total_val_gt_pairs = sum(len(gt[sid]) for sid in val_ids)
    found_val_gt_pairs = int(y_val.sum())
    print(f"Blocker Recall on Validation: {found_val_gt_pairs}/{total_val_gt_pairs} ({found_val_gt_pairs/total_val_gt_pairs*100:.2f}%)")
    
    # Load model
    model_data = joblib.load(config.OUTPUT_DIR / "code1_lambdarank.joblib")
    ranker = model_data["model"]
    val_scores = ranker.predict(x_val)
    
    # Baseline single cutoff
    truth_val = {sid: gt.get(sid, set()) for sid in val_ids}
    
    print("\n--- Strategy 1: Standard Single Threshold Sweep ---")
    best_f05, best_t, best_p, best_r = 0, 0, 0, 0
    for q in np.linspace(0.80, 0.999, 100):
        t = float(np.quantile(val_scores, q))
        pred = defaultdict(set)
        idx = 0
        for group in val_pair_groups:
            for sid, cid in group:
                if val_scores[idx] >= t:
                    pred[sid].add(cid)
                idx += 1
        pred_comp = {sid: pred.get(sid, set()) for sid in val_ids}
        f05, p, r = evaluate_macro_f05(truth_val, pred_comp)
        if f05 > best_f05:
            best_f05, best_t, best_p, best_r = f05, t, p, r
    print(f"Single Threshold Best: F0.5={best_f05:.4f}, P={best_p:.4f}, R={best_r:.4f} at t={best_t:.3f}")

    print("\n--- Strategy 2: Source-Aware (S2 vs S3) Separate Selection ---")
    # For each entity, evaluate candidates by source
    # Test grid of t_s2 and t_s3
    best_f05_2, best_t2, best_t3 = 0, 0, 0
    t_candidates = np.quantile(val_scores, np.linspace(0.85, 0.995, 20))
    for t2 in t_candidates:
        for t3 in t_candidates:
            pred = defaultdict(set)
            idx = 0
            for group in val_pair_groups:
                for sid, cid in group:
                    score = val_scores[idx]
                    if cid.startswith("S2-") and score >= t2:
                        pred[sid].add(cid)
                    elif cid.startswith("S3-") and score >= t3:
                        pred[sid].add(cid)
                    idx += 1
            pred_comp = {sid: pred.get(sid, set()) for sid in val_ids}
            f05, p, r = evaluate_macro_f05(truth_val, pred_comp)
            if f05 > best_f05_2:
                best_f05_2, best_t2, best_t3, best_p2, best_r2 = f05, t2, t3, p, r
    print(f"Source-Aware Best: F0.5={best_f05_2:.4f}, P={best_p2:.4f}, R={best_r2:.4f} at t2={best_t2:.3f}, t3={best_t3:.3f}")

    print("\n--- Strategy 3: Rank-Order + Source Top-1 with Relative Margin ---")
    # In each group:
    # Separate candidates into S2 and S3 lists sorted by score
    # Top S2 gets accepted if score >= base_t and (len == 1 or score - runner_up >= margin)
    for base_t in np.quantile(val_scores, [0.85, 0.90, 0.93, 0.95, 0.97]):
        for margin in [0.0, 0.2, 0.5, 0.8, 1.2]:
            pred = defaultdict(set)
            idx = 0
            for group in val_pair_groups:
                s2_cands = []
                s3_cands = []
                for sid, cid in group:
                    score = val_scores[idx]
                    if cid.startswith("S2-"):
                        s2_cands.append((cid, score))
                    else:
                        s3_cands.append((cid, score))
                    idx += 1
                s2_cands.sort(key=lambda x: x[1], reverse=True)
                s3_cands.sort(key=lambda x: x[1], reverse=True)
                
                # S2 decisions
                if s2_cands and s2_cands[0][1] >= base_t:
                    top_score = s2_cands[0][1]
                    pred[group[0][0]].add(s2_cands[0][0])
                    # secondary matches in S2 (if multi-match in S2)
                    for cid, sc in s2_cands[1:]:
                        if sc >= base_t and (top_score - sc) <= 0.3:
                            pred[group[0][0]].add(cid)
                
                # S3 decisions
                if s3_cands and s3_cands[0][1] >= base_t:
                    top_score = s3_cands[0][1]
                    pred[group[0][0]].add(s3_cands[0][0])
                    for cid, sc in s3_cands[1:]:
                        if sc >= base_t and (top_score - sc) <= 0.3:
                            pred[group[0][0]].add(cid)

            pred_comp = {sid: pred.get(sid, set()) for sid in val_ids}
            f05, p, r = evaluate_macro_f05(truth_val, pred_comp)
            if f05 > best_f05:
                print(f"  * New Peak Rank-Order: F0.5={f05:.4f}, P={p:.4f}, R={r:.4f} (base_t={base_t:.2f}, margin={margin})")

if __name__ == "__main__":
    run()
