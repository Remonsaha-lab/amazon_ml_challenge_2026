"""
Decision Layer & Competition Macro F0.5 Metric Optimization Module.
Directly implements the official competition evaluation metric including
singleton credit (1.0 vs 0.0), precision weighting (2x), and fine-grained
threshold search across tau in [0.50, 0.99].
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from collections import defaultdict


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """
    Compute official F0.5 score for a single Source 1 entity.
    - If true singleton and predict empty: 1.0
    - If true singleton and predict false match: 0.0
    - If non-empty true matches and predict empty: 0.0
    - Otherwise: F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    n_true = len(true_matches)
    n_pred = len(pred_matches)
    
    # Singleton logic
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
        
    if n_pred == 0:
        return 0.0
        
    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0
        
    precision = tp / n_pred
    recall = tp / n_true
    
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
        
    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    gt_map: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Tuple[float, float, float]:
    """
    Compute macro-averaged F0.5, mean precision, and mean recall across all S1 entities.
    """
    f05_scores = []
    precisions = []
    recalls = []
    
    for s1_id, true_set in gt_map.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        f05_scores.append(score)
        
        # Diagnostics
        if len(pred_set) > 0 and len(true_set) > 0:
            tp = len(true_set & pred_set)
            precisions.append(tp / len(pred_set))
            recalls.append(tp / len(true_set))
        elif len(pred_set) == 0 and len(true_set) == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            
    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    mean_p = float(np.mean(precisions)) if precisions else 0.0
    mean_r = float(np.mean(recalls)) if recalls else 0.0
    return macro_f05, mean_p, mean_r


def find_optimal_threshold(
    gt_map: Dict[str, Set[str]],
    pair_scores: List[Tuple[str, str, float]],
    val_s1_ids: Set[str],
    tau_range: np.ndarray = None
) -> Tuple[float, float, Dict[float, float]]:
    """
    Grid-search threshold tau to directly maximize Macro F0.5 on validation split.
    """
    if tau_range is None:
        tau_range = np.arange(0.50, 0.99, 0.01)
        
    # Group candidate predictions by S1 entity
    cand_by_s1 = defaultdict(list)
    for s1_id, cand_id, prob in pair_scores:
        if s1_id in val_s1_ids:
            cand_by_s1[s1_id].append((cand_id, prob))
            
    best_tau = 0.75
    best_f05 = -1.0
    curve = {}
    
    for tau in tau_range:
        tau = round(float(tau), 3)
        preds = {}
        for s1_id in val_s1_ids:
            cands = cand_by_s1.get(s1_id, [])
            matched = {cid for cid, p in cands if p >= tau}
            preds[s1_id] = matched
            
        macro_f05, _, _ = evaluate_macro_f05(
            {s1: gt_map.get(s1, set()) for s1 in val_s1_ids},
            preds
        )
        curve[tau] = macro_f05
        
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_tau = tau
            
    return best_tau, best_f05, curve
