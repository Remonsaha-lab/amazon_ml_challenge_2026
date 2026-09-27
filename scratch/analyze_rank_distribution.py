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
from rank_pipeline import read_source, get_s1_records, collect_training_candidates, make_rank_matrix

def main():
    print("Loading truth...")
    full_gt = load_ground_truth_map(config.TRAIN_GROUND_TRUTH)
    all_ids = list(full_gt)
    rng = random.Random(42)
    cohort_ids = rng.sample(all_ids, 2000)
    val_ids = set(cohort_ids[1500:])
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

    offset = 0
    s2_ranks = []
    s3_ranks = []
    for group in val_pair_groups:
        sid = group[0][0]
        true_matches = gt.get(sid, set())
        
        group_s2 = []
        group_s3 = []
        for s_id, c_id in group:
            sc = val_scores[offset]
            if c_id.startswith('S2-'):
                group_s2.append((c_id, sc))
            else:
                group_s3.append((c_id, sc))
            offset += 1
        
        group_s2.sort(key=lambda x: x[1], reverse=True)
        group_s3.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (c_id, sc) in enumerate(group_s2, 1):
            if c_id in true_matches:
                s2_ranks.append(rank)
        for rank, (c_id, sc) in enumerate(group_s3, 1):
            if c_id in true_matches:
                s3_ranks.append(rank)

    print(f"S2 True Matches at Rank 1: {sum(r == 1 for r in s2_ranks) / len(s2_ranks) * 100:.2f}% ({sum(r == 1 for r in s2_ranks)}/{len(s2_ranks)})")
    print(f"S3 True Matches at Rank 1: {sum(r == 1 for r in s3_ranks) / len(s3_ranks) * 100:.2f}% ({sum(r == 1 for r in s3_ranks)}/{len(s3_ranks)})")
    print(f"S2 True Matches in Top 2:  {sum(r <= 2 for r in s2_ranks) / len(s2_ranks) * 100:.2f}% ({sum(r <= 2 for r in s2_ranks)}/{len(s2_ranks)})")
    print(f"S3 True Matches in Top 2:  {sum(r <= 2 for r in s3_ranks) / len(s3_ranks) * 100:.2f}% ({sum(r <= 2 for r in s3_ranks)}/{len(s3_ranks)})")

if __name__ == "__main__":
    main()
