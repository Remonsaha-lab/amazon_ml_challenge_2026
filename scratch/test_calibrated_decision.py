import sys, joblib, random
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'code_1'))
sys.path.insert(0, str(ROOT / 'code' / 'business_entity_resolution' / 'src'))

import config
from dataset import load_ground_truth_map, RecordCache
from blocking import MultiKeyBlocker
from decision import evaluate_macro_f05
from rank_pipeline import read_source, get_s1_records, collect_training_candidates, make_rank_matrix

def main():
    print("Loading truth and running validation check...")
    full_gt = load_ground_truth_map(config.TRAIN_GROUND_TRUTH)
    all_ids = list(full_gt)
    rng = random.Random(42)
    cohort_ids = rng.sample(all_ids, 4000)
    val_ids = set(cohort_ids[3000:])
    gt = {sid: full_gt[sid] for sid in cohort_ids}
    val_s1 = get_s1_records(config.TRAIN_SOURCE1, val_ids)

    candidate_records = collect_training_candidates(set(cohort_ids), gt, 50000)
    cache = RecordCache(candidate_records)
    blocker = MultiKeyBlocker(max_candidates_per_s1=100)
    blocker.build_candidate_index_from_records(candidate_records, show_progress=False)

    x_val, y_val, group_val, val_pair_groups = make_rank_matrix(val_s1, cache, blocker, gt, 100, keep_pairs=True)

    model_data = joblib.load(config.OUTPUT_DIR / 'code1_lambdarank.joblib')
    ranker = model_data['model']
    val_scores = ranker.predict(x_val)
    truth_val = {sid: gt.get(sid, set()) for sid in val_ids}

    # Baseline: single threshold
    baseline_t = model_data['threshold']
    pred_base = defaultdict(set)
    idx = 0
    for group in val_pair_groups:
        for sid, cid in group:
            if val_scores[idx] >= baseline_t:
                pred_base[sid].add(cid)
            idx += 1
    f05_b, p_b, r_b = evaluate_macro_f05(truth_val, {sid: pred_base.get(sid, set()) for sid in val_ids})
    print(f"Original Saved Model Cutoff ({baseline_t:.3f}): F0.5={f05_b:.4f}, P={p_b:.4f}, R={r_b:.4f}")

    # Advanced Multi-Source Confidence Decision Rule
    print("\nGrid searching Source-Aware Calibrated Decision Engine...")
    best_f05 = f05_b
    best_params = None

    # Test combination of base cutoff for top candidate, runner-up margin, and minimum score
    for tau_top in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]:
        for tau_runner_up in [0.8, 1.2, 1.5, 1.8, 2.0]:
            for margin in [0.2, 0.35, 0.5, 0.8]:
                pred = defaultdict(set)
                idx = 0
                for group in val_pair_groups:
                    sid = group[0][0]
                    s2_cands = []
                    s3_cands = []
                    for s_id, c_id in group:
                        sc = float(val_scores[idx])
                        if c_id.startswith("S2-"):
                            s2_cands.append((c_id, sc))
                        else:
                            s3_cands.append((c_id, sc))
                        idx += 1
                    s2_cands.sort(key=lambda x: x[1], reverse=True)
                    s3_cands.sort(key=lambda x: x[1], reverse=True)

                    # For S2:
                    if s2_cands and s2_cands[0][1] >= tau_top:
                        top_sc = s2_cands[0][1]
                        pred[sid].add(s2_cands[0][0])
                        for cid, sc in s2_cands[1:]:
                            if sc >= tau_runner_up and (top_sc - sc) <= margin:
                                pred[sid].add(cid)

                    # For S3:
                    if s3_cands and s3_cands[0][1] >= tau_top:
                        top_sc = s3_cands[0][1]
                        pred[sid].add(s3_cands[0][0])
                        for cid, sc in s3_cands[1:]:
                            if sc >= tau_runner_up and (top_sc - sc) <= margin:
                                pred[sid].add(cid)

                pred_comp = {sid: pred.get(sid, set()) for sid in val_ids}
                f05, p, r = evaluate_macro_f05(truth_val, pred_comp)
                if f05 > best_f05:
                    best_f05 = f05
                    best_params = (tau_top, tau_runner_up, margin, p, r)
                    print(f"  * UPGRADE FOUND: F0.5={f05:.4f}, P={p:.4f}, R={r:.4f} (tau_top={tau_top}, tau_runner={tau_runner_up}, margin={margin})")

    if best_params:
        print(f"\nFinal Best Result: F0.5={best_f05:.4f}, P={best_params[3]:.4f}, R={best_params[4]:.4f}")

if __name__ == "__main__":
    main()
