"""
Decision Layer & Competition Macro F0.5 Metric Optimization Module.
Directly implements the official competition evaluation metric including
singleton credit (1.0 vs 0.0), precision weighting (2x), fine-grained
threshold search across tau in [0.50, 0.98] with step 0.005, and
calibrated top-candidate margin decision logic.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
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


def filter_candidates_with_margin(
    candidates_with_probs: List[Any],
    tau: float,
    high_conf_tau: float = 0.88,
    max_margin_drop: float = 0.15
) -> List[str]:
    """
    Calibrated decision-layer filtering with margin and confidence tiers:
    - If no candidate >= tau: singleton (empty)
    - High confidence (p >= high_conf_tau): accept (unless extreme PIN conflict with negligible address match).
    - Medium confidence (tau <= p < high_conf_tau):
      * Requires margin from best candidate <= max_margin_drop (prevents ambiguous trailing matches)
      * Rejects candidates with conflicting house numbers or conflicting PINs + weak address match.
    """
    valid = []
    for item in candidates_with_probs:
        if len(item) >= 5:
            cid, p, addr_ratio, pin_mismatch, house_mismatch = item[0], float(item[1]), float(item[2]), float(item[3]), float(item[4])
        elif len(item) == 2:
            cid, p = item[0], float(item[1])
            addr_ratio, pin_mismatch, house_mismatch = 1.0, 0.0, 0.0
        else:
            cid, p = item[0], float(item[1])
            addr_ratio, pin_mismatch, house_mismatch = 1.0, 0.0, 0.0
            
        if p >= tau:
            valid.append((cid, p, addr_ratio, pin_mismatch, house_mismatch))
            
    if not valid:
        return []
    
    # Sort descending by model probability
    valid.sort(key=lambda x: x[1], reverse=True)
    p_best = valid[0][1]
    
    accepted = []
    for cid, p, addr_ratio, pin_mismatch, house_mismatch in valid:
        if p >= high_conf_tau:
            # Extreme conflict rejection: different PIN + near-zero address similarity
            if pin_mismatch == 1.0 and addr_ratio < 0.20:
                continue
            accepted.append(cid)
        else:
            # Medium confidence checks:
            # 1. Ambiguity margin drop relative to top candidate
            if (p_best - p) > max_margin_drop:
                continue
            # 2. Conflicting house/door number on street
            if house_mismatch == 1.0 and addr_ratio < 0.60:
                continue
            # 3. Conflicting postal PIN
            if pin_mismatch == 1.0 and addr_ratio < 0.40:
                continue
            accepted.append(cid)
            
    return accepted


def find_optimal_threshold(
    gt_map: Dict[str, Set[str]],
    pair_scores: List[Tuple[str, str, float]],
    val_s1_ids: Set[str],
    tau_range: np.ndarray = None
) -> Tuple[float, float, Dict[float, float]]:
    """
    Fine-grained grid-search threshold tau (step: 0.005) to directly maximize Macro F0.5 on validation split.
    """
    if tau_range is None:
        tau_range = np.arange(0.50, 0.995, 0.005)
        
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
            matched = set(filter_candidates_with_margin(cands, tau=tau))
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
