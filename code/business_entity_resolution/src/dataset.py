"""
Dataset Construction and Pairwise Sampling for Amazon ML Challenge 2026.
Creates entity-level grouped train/val splits, mines hard negatives from the
blocking candidate pool, and extracts 24-dimensional feature matrices.
"""

import os
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from normalization import (
    normalize_name,
    normalize_address,
    extract_postal_code,
    extract_numeric_tokens,
    normalize_country
)
from features import compute_pair_features, FEATURE_NAMES
from blocking import MultiKeyBlocker
import config


class RecordCache:
    """Pre-computes and caches normalized fields for high-speed feature extraction."""
    def __init__(self, df: pd.DataFrame):
        self.names = {}
        self.addrs = {}
        self.countries = {}
        self.norm_names = {}
        self.norm_addrs = {}
        self.pins = {}
        self.nums = {}

        for row in df.itertuples(index=False):
            eid = row.entity_id
            name = str(row.business_name or "")
            addr = str(row.business_address or "")
            cntry = str(row.country or "")

            self.names[eid] = name
            self.addrs[eid] = addr
            self.countries[eid] = cntry
            self.norm_names[eid] = normalize_name(name)
            self.norm_addrs[eid] = normalize_address(addr)
            self.pins[eid] = extract_postal_code(addr, cntry)
            self.nums[eid] = extract_numeric_tokens(addr)


def load_ground_truth_map(gt_path: str) -> Dict[str, Set[str]]:
    """Loads ground truth into {source1_entity_id: set_of_matched_ids}."""
    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    df_gt["matched_entity_ids"] = df_gt["matched_entity_ids"].fillna("")
    
    gt_map = {}
    for row in df_gt.itertuples(index=False):
        s1_id = row.source1_entity_id
        raw_m = str(row.matched_entity_ids or "").strip()
        targets = {x.strip() for x in raw_m.split(",") if x.strip()} if raw_m else set()
        gt_map[s1_id] = targets
    return gt_map


def build_pairwise_dataset(
    s1_cache: RecordCache,
    cand_cache: RecordCache,
    candidate_dict: Dict[str, Set[str]],
    gt_map: Dict[str, Set[str]],
    is_training: bool = True,
    hard_neg_ratio: int = 4,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Constructs feature matrix X, binary labels y, and (s1_id, cand_id) pair tuples.
    If is_training=True, subsamples hard negatives from candidate pool to maintain hard_neg_ratio.
    If is_training=False (validation), evaluates all candidate pairs.
    """
    rng = np.random.RandomState(random_seed)
    features_list = []
    labels_list = []
    pairs_list = []

    for s1_id, candidate_set in tqdm(candidate_dict.items(), desc="Extracting Pair Features"):
        true_matches = gt_map.get(s1_id, set())
        
        # Partition candidates into positives and negatives
        pos_cands = [c for c in candidate_set if c in true_matches]
        neg_cands = [c for c in candidate_set if c not in true_matches]

        # For training: ensure all known true matches in candidate set are included
        # and subsample negatives to control positive:negative ratio
        if is_training:
            # Also add true matches that might have been in ground truth but missed by candidate set if in cache
            for m in true_matches:
                if m in cand_cache.names and m not in pos_cands:
                    pos_cands.append(m)

            if not pos_cands:
                # If singleton or no positive matches in cache, sample a small negative set
                selected_negs = neg_cands[:2]
            else:
                max_negs = max(len(pos_cands) * hard_neg_ratio, 2)
                if len(neg_cands) > max_negs:
                    selected_negs = list(rng.choice(neg_cands, size=max_negs, replace=False))
                else:
                    selected_negs = neg_cands
            eval_pairs = [(c, 1.0) for c in pos_cands] + [(c, 0.0) for c in selected_negs]
        else:
            # For validation: evaluate all candidates produced by blocking
            eval_pairs = [(c, 1.0 if c in true_matches else 0.0) for c in candidate_set]

        # Extract features for each pair
        for cand_id, label in eval_pairs:
            feat_vec = compute_pair_features(
                s1_name=s1_cache.names[s1_id],
                s1_addr=s1_cache.addrs[s1_id],
                s1_country=s1_cache.countries[s1_id],
                cand_id=cand_id,
                cand_name=cand_cache.names[cand_id],
                cand_addr=cand_cache.addrs[cand_id],
                cand_country=cand_cache.countries[cand_id],
                s1_norm_name=s1_cache.norm_names[s1_id],
                s1_norm_addr=s1_cache.norm_addrs[s1_id],
                s1_pin=s1_cache.pins[s1_id],
                s1_nums=s1_cache.nums[s1_id],
                cand_norm_name=cand_cache.norm_names[cand_id],
                cand_norm_addr=cand_cache.norm_addrs[cand_id],
                cand_pin=cand_cache.pins[cand_id],
                cand_nums=cand_cache.nums[cand_id]
            )
            features_list.append(feat_vec)
            labels_list.append(label)
            pairs_list.append((s1_id, cand_id))

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.float32)
    return X, y, pairs_list
