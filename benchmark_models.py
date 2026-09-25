import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "code" / "business_entity_resolution"))

from normalization import normalize_country
from blocking import MultiKeyBlocker
from dataset import RecordCache, load_ground_truth_map, build_pairwise_dataset
from models import train_lightgbm, train_xgboost, get_feature_importances
from decision import find_optimal_threshold, evaluate_macro_f05
import config

print("=" * 80)
print("STEP 5: HEAD-TO-HEAD BENCHMARK: LIGHTGBM VS. XGBOOST")
print("Macro F0.5 Metric Optimization on Entity-Grouped Validation Split")
print("=" * 80)

# 1. Load Ground Truth Cohort
SAMPLE_S1_SIZE = 3000
print(f"\n1. Loading Ground Truth cohort ({SAMPLE_S1_SIZE} S1 entities)...")
gt_map_full = load_ground_truth_map(config.TRAIN_GROUND_TRUTH)

# Sample S1 entities (mix of matches and singletons)
s1_cohort_ids = list(gt_map_full.keys())[:SAMPLE_S1_SIZE]
gt_map = {sid: gt_map_full[sid] for sid in s1_cohort_ids}

# 2. Entity-level Grouped Train/Validation Split (80% train / 20% val)
train_s1_ids, val_s1_ids = train_test_split(s1_cohort_ids, test_size=0.20, random_state=config.RANDOM_SEED)
train_s1_set = set(train_s1_ids)
val_s1_set = set(val_s1_ids)
print(f"   Train S1 Entities: {len(train_s1_set):,}")
print(f"   Val S1 Entities  : {len(val_s1_set):,}")

# 3. Load S1 Records
print("\n2. Loading S1 entity records...")
s1_rows = []
cohort_set = set(s1_cohort_ids)
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

# 4. Load Candidate Records (all target matches + background candidate pool)
print("\n3. Loading candidate pool from Source 2 & Source 3...")
all_target_ids = set()
for sid in s1_cohort_ids:
    all_target_ids.update(gt_map[sid])

cand_rows = []
BACKGROUND_POOL = 80000
c2 = c3 = 0

with open(config.TRAIN_SOURCE2, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            if parts[0] in all_target_ids or c2 < (BACKGROUND_POOL // 2):
                cand_rows.append(parts)
                c2 += 1

with open(config.TRAIN_SOURCE3, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            if parts[0] in all_target_ids or c3 < (BACKGROUND_POOL // 2):
                cand_rows.append(parts)
                c3 += 1

df_cands = pd.DataFrame(cand_rows, columns=["entity_id", "business_name", "business_address", "country"])
print(f"   Total Candidate Records Loaded: {len(df_cands):,}")
cand_cache = RecordCache(df_cands)

# 5. Candidate Generation via MultiKeyBlocker
print("\n4. Indexing candidates and generating candidate sets...")
blocker = MultiKeyBlocker(max_candidates_per_s1=config.MAX_CANDIDATES_PER_S1)
blocker.build_candidate_index(df_cands, show_progress=False)

s1_train_df = df_s1[df_s1["entity_id"].isin(train_s1_set)]
s1_val_df = df_s1[df_s1["entity_id"].isin(val_s1_set)]

cand_train = blocker.generate_candidate_pairs(s1_train_df, show_progress=False)
cand_val = blocker.generate_candidate_pairs(s1_val_df, show_progress=False)

# 6. Extract Pairwise Features
print("\n5. Extracting Pairwise Features for Training & Validation splits...")
X_train, y_train, train_pairs = build_pairwise_dataset(
    s1_cache, cand_cache, cand_train, gt_map, is_training=True, hard_neg_ratio=4
)
X_val, y_val, val_pairs = build_pairwise_dataset(
    s1_cache, cand_cache, cand_val, gt_map, is_training=False
)

print(f"   X_train shape: {X_train.shape}, Positives: {int(y_train.sum()):,}, Negatives: {int(len(y_train) - y_train.sum()):,}")
print(f"   X_val shape  : {X_val.shape}, Positives: {int(y_val.sum()):,}, Negatives: {int(len(y_val) - y_val.sum()):,}")

# 7. Model A: Train LightGBM
print("\n6. Training Model A: LightGBM Classifier...")
lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)
lgbm_val_probs = lgbm_model.predict_proba(X_val)[:, 1]

val_pair_scores_lgb = [
    (val_pairs[i][0], val_pairs[i][1], float(lgbm_val_probs[i]))
    for i in range(len(val_pairs))
]
tau_lgb, f05_lgb, curve_lgb = find_optimal_threshold(gt_map, val_pair_scores_lgb, val_s1_set)

# 8. Model B: Train XGBoost
print("\n7. Training Model B: XGBoost Classifier...")
xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]

val_pair_scores_xgb = [
    (val_pairs[i][0], val_pairs[i][1], float(xgb_val_probs[i]))
    for i in range(len(val_pairs))
]
tau_xgb, f05_xgb, curve_xgb = find_optimal_threshold(gt_map, val_pair_scores_xgb, val_s1_set)

# 9. Comparison & Final Verdict
print("\n" + "=" * 80)
print("HEAD-TO-HEAD BENCHMARK RESULTS")
print("=" * 80)
print(f"LightGBM -> Best Threshold: {tau_lgb:.2f} | Validation Macro F0.5: {f05_lgb:.4f}")
print(f"XGBoost  -> Best Threshold: {tau_xgb:.2f} | Validation Macro F0.5: {f05_xgb:.4f}")

# Top 8 Feature Importances for Best Model
best_model = lgbm_model if f05_lgb >= f05_xgb else xgb_model
best_name = "LightGBM" if f05_lgb >= f05_xgb else "XGBoost"
print(f"\nWinning Model: {best_name}")
print("\nTop 8 Most Predictive Features:")
importances = get_feature_importances(best_model)
for f_name, imp in list(importances.items())[:8]:
    print(f"  {f_name:25s}: {imp * 100:6.2f}%")

print("=" * 80)
