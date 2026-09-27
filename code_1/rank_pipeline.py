"""Query-wise LambdaRank alternative for business entity resolution.

Unlike the pairwise classifier in code/, this model trains on candidate lists
grouped by each source-1 record and learns which candidates should rank first.
It reuses the repository's normalization, blocking and pair-feature utilities.
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRanker, early_stopping, log_evaluation


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

import config
from blocking import MultiKeyBlocker
from dataset import RecordCache, load_ground_truth_map
from decision import evaluate_macro_f05
from features import compute_pair_features, FEATURE_NAMES
from normalization import (
    extract_numeric_tokens,
    extract_postal_code,
    is_empty_or_nan,
    normalize_address,
    normalize_country,
    normalize_name,
)

Record = Tuple[str, str, str, str]


def read_source(path: Path):
    with path.open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4:
                yield (p[0].strip(), p[1].strip(), p[2].strip(), p[3].strip())


def get_s1_records(path: Path, wanted: Set[str]) -> List[Record]:
    found = {}
    for rec in read_source(path):
        if rec[0] in wanted:
            found[rec[0]] = rec
            if len(found) == len(wanted):
                break
    return list(found.values())


def collect_training_candidates(cohort: Set[str], gt: Dict[str, Set[str]], max_background: int):
    needed = set().union(*(gt.get(sid, set()) for sid in cohort)) if cohort else set()
    records = []
    per_source_bg = max_background // 2
    for path in (config.TRAIN_SOURCE2, config.TRAIN_SOURCE3):
        bg = 0
        for rec in read_source(path):
            if rec[0] in needed:
                records.append(rec)
            elif bg < per_source_bg:
                records.append(rec)
                bg += 1
    return records


def make_rank_matrix(
    s1_records: List[Record], cache: RecordCache, blocker: MultiKeyBlocker,
    gt: Dict[str, Set[str]], max_cands: int, keep_pairs: bool = False,
):
    capacity = len(s1_records) * max_cands
    x_matrix = np.empty((capacity, len(FEATURE_NAMES)), dtype=np.float32)
    y_vector = np.empty(capacity, dtype=np.int32)
    used = 0
    groups, pair_groups = [], []
    for sid, name, address, country in s1_records:
        cands = blocker.retrieve_candidates_for_record(sid, name, address, country)[:max_cands]
        s1_name = normalize_name(name)
        s1_address = normalize_address(address)
        s1_pin = extract_postal_code(address, country)
        s1_numbers = extract_numeric_tokens(address)
        group_x, group_y, group_pairs = [], [], []
        for cid in cands:
            if cid not in cache.norm_names:
                continue
            group_x.append(compute_pair_features(
                "", "", country, cid, "", "", cache.countries.get(cid, ""),
                s1_name, s1_address, s1_pin, s1_numbers,
                cache.norm_names[cid], cache.norm_addrs[cid], cache.pins[cid], cache.nums[cid],
            ))
            group_y.append(1 if cid in gt.get(sid, set()) else 0)
            group_pairs.append((sid, cid))
        if group_x:
            count = len(group_x)
            x_matrix[used:used + count] = np.asarray(group_x, dtype=np.float32)
            y_vector[used:used + count] = group_y
            used += count
            groups.append(len(group_x))
            if keep_pairs:
                pair_groups.append(group_pairs)
    return x_matrix[:used], y_vector[:used], groups, pair_groups


def flatten_pair_groups(pair_groups):
    return [pair for group in pair_groups for pair in group]


def select_group_matches(
    group: List[Tuple[str, str]],
    scores: np.ndarray,
    offset: int,
    tau_top_s2: float,
    tau_top_s3: float,
    tau_runner: float,
    margin: float
) -> Tuple[List[str], int]:
    """
    Source-Aware Top-K Decision Rule for multi-source entity resolution:
    - 80.5% of true entities have matches in both S2 and S3.
    - Evaluates S2 candidates and S3 candidates separately with dedicated source cutoffs.
    - Recovers secondary intra-source matches only when candidate score is within 'margin' of top score.
    """
    s2_cands = []
    s3_cands = []
    for sid, cid in group:
        sc = float(scores[offset])
        if cid.startswith("S2-"):
            s2_cands.append((cid, sc))
        else:
            s3_cands.append((cid, sc))
        offset += 1
    
    s2_cands.sort(key=lambda x: x[1], reverse=True)
    s3_cands.sort(key=lambda x: x[1], reverse=True)
    
    chosen = []
    # S2 selection
    if s2_cands and s2_cands[0][1] >= tau_top_s2:
        top_sc = s2_cands[0][1]
        chosen.append(s2_cands[0][0])
        for cid, sc in s2_cands[1:]:
            if sc >= tau_runner and (top_sc - sc) <= margin:
                chosen.append(cid)
                
    # S3 selection
    if s3_cands and s3_cands[0][1] >= tau_top_s3:
        top_sc = s3_cands[0][1]
        chosen.append(s3_cands[0][0])
        for cid, sc in s3_cands[1:]:
            if sc >= tau_runner and (top_sc - sc) <= margin:
                chosen.append(cid)
                
    return chosen, offset


def tune_threshold(gt: Dict[str, Set[str]], val_ids: Set[str], pair_groups, scores):
    all_scores = np.asarray(scores, dtype=np.float64)
    if not len(all_scores):
        return {"tau_top_s2": 1.0, "tau_top_s3": 1.0, "tau_runner": 1.2, "margin": 0.5}, 0.0, {}
    
    truth_val = {sid: gt.get(sid, set()) for sid in val_ids}
    
    # 1. Sweep scalar thresholds across fine percentiles
    q_candidates = np.unique(np.quantile(all_scores, np.linspace(0.80, 0.999, 50)))
    best_params = {}
    best_f05 = -1.0
    best_metrics = {}
    
    for t in q_candidates:
        pred = defaultdict(set)
        idx = 0
        for group in pair_groups:
            sid = group[0][0]
            for _, cid in group:
                if all_scores[idx] >= t:
                    pred[sid].add(cid)
                idx += 1
        pred_complete = {sid: pred.get(sid, set()) for sid in val_ids}
        f05, prec, rec = evaluate_macro_f05(truth_val, pred_complete)
        if f05 > best_f05:
            exact_acc = sum(pred_complete[sid] == truth_val[sid] for sid in val_ids) / max(len(val_ids), 1)
            best_f05 = f05
            best_params = {"tau_top_s2": float(t), "tau_top_s3": float(t), "tau_runner": float(t) + 0.3, "margin": 0.4}
            best_metrics = {"macro_f0_5": f05, "mean_precision": prec, "mean_recall": rec, "exact_set_accuracy": exact_acc}
            
    # 2. Refine with source-aware top-1 and margin grid search
    t_center = best_params["tau_top_s2"]
    fine_cutoffs = np.linspace(max(t_center - 0.6, -1.0), t_center + 0.6, 7)
    
    for t_top_s2 in fine_cutoffs:
        for t_top_s3 in fine_cutoffs:
            for margin in [0.25, 0.4, 0.6]:
                tau_runner = max(t_top_s2, t_top_s3) + 0.2
                pred = defaultdict(set)
                offset = 0
                for group in pair_groups:
                    sid = group[0][0]
                    chosen, offset = select_group_matches(
                        group, all_scores, offset, t_top_s2, t_top_s3, tau_runner, margin
                    )
                    pred[sid] = set(chosen)
                pred_complete = {sid: pred.get(sid, set()) for sid in val_ids}
                f05, prec, rec = evaluate_macro_f05(truth_val, pred_complete)
                if f05 > best_f05:
                    exact_acc = sum(pred_complete[sid] == truth_val[sid] for sid in val_ids) / max(len(val_ids), 1)
                    best_f05 = f05
                    best_params = {"tau_top_s2": float(t_top_s2), "tau_top_s3": float(t_top_s3),
                                   "tau_runner": float(tau_runner), "margin": float(margin)}
                    best_metrics = {"macro_f0_5": f05, "mean_precision": prec, "mean_recall": rec, "exact_set_accuracy": exact_acc}
                    
    return best_params, best_f05, best_metrics


def train(args):
    full_gt = load_ground_truth_map(config.TRAIN_GROUND_TRUTH)
    all_ids = list(full_gt)
    if args.train_cohort and args.train_cohort < len(all_ids):
        rng = random.Random(args.seed)
        all_ids = rng.sample(all_ids, args.train_cohort)
    
    # Stratified/random split, capping validation cohort to 6,000 for fast threshold tuning
    if len(all_ids) > 6000:
        val_count = min(int(len(all_ids) * args.validation_size), 6000)
        rng = random.Random(args.seed)
        rng.shuffle(all_ids)
        val_ids_list = all_ids[:val_count]
        train_ids_list = all_ids[val_count:]
    else:
        train_ids_list, val_ids_list = train_test_split(all_ids, test_size=args.validation_size, random_state=args.seed)

    train_ids, val_ids = set(train_ids_list), set(val_ids_list)
    cohort = train_ids | val_ids
    gt = {sid: full_gt[sid] for sid in cohort}
    print(f"Training ranking model with {len(train_ids):,} S1 groups; validating with {len(val_ids):,}.", flush=True)

    train_s1 = get_s1_records(config.TRAIN_SOURCE1, train_ids)
    val_s1 = get_s1_records(config.TRAIN_SOURCE1, val_ids)
    bg = min(max(len(cohort) * 4, 100_000), args.background_cap)
    candidate_records = collect_training_candidates(cohort, gt, bg)
    print(f"Indexed {len(candidate_records):,} candidate records.", flush=True)
    cache = RecordCache(candidate_records)
    blocker = MultiKeyBlocker(max_candidates_per_s1=args.max_candidates)
    blocker.build_candidate_index_from_records(candidate_records, show_progress=False)

    x_train, y_train, group_train, _ = make_rank_matrix(train_s1, cache, blocker, gt, args.max_candidates)
    x_val, y_val, group_val, val_pair_groups = make_rank_matrix(
        val_s1, cache, blocker, gt, args.max_candidates, keep_pairs=True
    )
    print(f"Rank matrices: train={x_train.shape}, validation={x_val.shape}; positive pairs={int(y_train.sum()):,}.", flush=True)

    ranker = LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=args.n_estimators,
        learning_rate=0.04, num_leaves=31, max_depth=-1, min_child_samples=30,
        reg_lambda=2.0, random_state=args.seed, n_jobs=-1, verbosity=-1,
        label_gain=[0, 3],
    )
    ranker.fit(
        x_train, y_train, group=group_train,
        eval_set=[(x_val, y_val)], eval_group=[group_val],
        callbacks=[early_stopping(40, verbose=False), log_evaluation(50)],
    )
    val_scores = ranker.predict(x_val)
    threshold, _, metrics = tune_threshold(gt, val_ids, val_pair_groups, val_scores)
    print("Held-out metrics at selected cutoff:", metrics, flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    import joblib
    model_path = args.output / "code1_lambdarank.joblib"
    joblib.dump({"model": ranker, "threshold": threshold, "max_candidates": args.max_candidates}, model_path)
    blocker.clear()
    cache.clear()
    del x_train, y_train, x_val, y_val, candidate_records
    gc.collect()
    return ranker, threshold


def make_test_features(record: Record, cands: List[str], cache: RecordCache):
    _, name, address, country = record
    n, a = normalize_name(name), normalize_address(address)
    pin, nums = extract_postal_code(address, country), extract_numeric_tokens(address)
    ids, vectors = [], []
    for cid in cands:
        if cid in cache.norm_names:
            ids.append(cid)
            vectors.append(compute_pair_features(
                "", "", country, cid, "", "", cache.countries.get(cid, ""),
                n, a, pin, nums, cache.norm_names[cid], cache.norm_addrs[cid],
                cache.pins[cid], cache.nums[cid],
            ))
    return ids, np.asarray(vectors, dtype=np.float32) if vectors else np.empty((0, len(FEATURE_NAMES)), np.float32)


def infer(ranker, threshold, args):
    print("\n[Test Inference] Reading test S1 entities...", flush=True)
    s1_by_country = defaultdict(list)
    s1_all_ids = []
    for rec in read_source(config.TEST_SOURCE1):
        s1_by_country[normalize_country(rec[3])].append(rec)
        s1_all_ids.append(rec[0])

    match_path = args.output / "matching_results.tsv"
    cand_path = args.output / "candidate_pairs.tsv"
    args.output.mkdir(parents=True, exist_ok=True)

    master_matches = {}
    master_candidates = {}

    batch_size = 1000

    for country, source1 in s1_by_country.items():
        print(f"\nIndexing {country} candidate partition ({len(source1):,} S1 queries)...", flush=True)
        candidate_records = []
        for path in (config.TEST_SOURCE2, config.TEST_SOURCE3):
            candidate_records.extend(rec for rec in read_source(path) if normalize_country(rec[3]) == country)
        
        cache = RecordCache(candidate_records)
        blocker = MultiKeyBlocker(max_candidates_per_s1=args.max_candidates)
        blocker.build_candidate_index_from_records(candidate_records, show_progress=False)
        print(f"   Indexed {len(candidate_records):,} candidates for {country}. Scoring S1 queries in batches...", flush=True)

        for b_start in range(0, len(source1), batch_size):
            b_end = min(b_start + batch_size, len(source1))
            batch_s1 = source1[b_start:b_end]

            batch_info = []
            batch_matrices = []

            for rec in batch_s1:
                sid = rec[0]
                cands = blocker.retrieve_candidates_for_record(rec[0], rec[1], rec[2], rec[3])[:args.max_candidates]
                cand_str = ",".join(cands)
                master_candidates[sid] = cand_str

                if not cands:
                    batch_info.append((sid, [], 0, 0))
                    continue

                ids, matrix = make_test_features(rec, cands, cache)
                feat_start = len(batch_matrices)
                if len(ids):
                    batch_matrices.append(matrix)
                batch_info.append((sid, ids, feat_start, len(ids)))

            # Batched prediction
            if batch_matrices:
                stacked_X = np.vstack(batch_matrices)
                all_scores = ranker.predict(stacked_X)
            else:
                all_scores = np.empty(0, dtype=np.float32)

            # Apply thresholds
            score_offset = 0
            for sid, ids, feat_start, feat_len in batch_info:
                if feat_len == 0:
                    master_matches[sid] = ""
                else:
                    scores = all_scores[score_offset : score_offset + feat_len]
                    score_offset += feat_len

                    if isinstance(threshold, dict):
                        dummy_group = [(sid, cid) for cid in ids]
                        chosen, _ = select_group_matches(
                            dummy_group, scores, 0,
                            threshold["tau_top_s2"], threshold["tau_top_s3"],
                            threshold["tau_runner"], threshold["margin"]
                        )
                    else:
                        chosen = [cid for cid, score in zip(ids, scores) if score >= threshold]
                    master_matches[sid] = ",".join(chosen)

        blocker.clear()
        cache.clear()
        del candidate_records, blocker, cache
        gc.collect()

    # Master Row-Preserved Output Writing
    print(f"\n[Assembly] Writing final submission files matching test_source1.tsv row order...", flush=True)
    total_written = 0
    non_empty = 0
    with match_path.open("w", encoding="utf-8", newline="") as fm, cand_path.open("w", encoding="utf-8", newline="") as fc:
        wm, wc = csv.writer(fm, delimiter="\t", lineterminator="\n"), csv.writer(fc, delimiter="\t", lineterminator="\n")
        wm.writerow(("source1_entity_id", "matched_entity_ids"))
        wc.writerow(("source1_entity_id", "candidate_entity_ids"))

        for sid in s1_all_ids:
            m = master_matches.get(sid, "")
            c = master_candidates.get(sid, "")
            wm.writerow((sid, m))
            wc.writerow((sid, c))
            total_written += 1
            if m:
                non_empty += 1

    print(f"Wrote {match_path} and {cand_path}.", flush=True)
    print(f"Total Rows: {total_written:,} (100% coverage, Non-empty: {non_empty:,}). Safe to submit!", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Group-wise LightGBM LambdaRank entity matcher")
    parser.add_argument("--train-cohort", type=int, default=30_000, help="0 means all labeled S1 rows")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--background-cap", type=int, default=400_000)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--train-only", action="store_true", help="Train and validate without test inference")
    parser.add_argument("--infer-only", action="store_true", help="Run test inference using existing saved model")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to saved joblib model")
    args = parser.parse_args()

    import joblib
    saved_model_file = args.model_path or (args.output / "code1_lambdarank.joblib")

    if args.infer_only:
        if not saved_model_file.is_file():
            raise SystemExit(f"Model file not found: {saved_model_file}. Train first or specify --model-path.")
        print(f"Loading existing model from {saved_model_file}...", flush=True)
        data = joblib.load(saved_model_file)
        model = data["model"]
        threshold = data["threshold"]
        print(f"Loaded model successfully. Threshold: {threshold}", flush=True)
        infer(model, threshold, args)
        return

    model, threshold = train(args)
    if not args.train_only:
        infer(model, threshold, args)


if __name__ == "__main__":
    main()

