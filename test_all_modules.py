"""
ML Challenge 2026: Comprehensive Module-by-Module Verification Suite.
Tests every file in code/business_entity_resolution/src/ individually
with assertions and performance checks.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Ensure UTF-8 stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

passed = 0
failed = 0

def report(module_name: str, test_name: str, success: bool, msg: str = ""):
    global passed, failed
    status = "PASS" if success else "FAIL"
    icon = "[OK]" if success else "[X]"
    print(f"  {icon} [{status}] {module_name} :: {test_name} {f'({msg})' if msg else ''}")
    if success:
        passed += 1
    else:
        failed += 1


print("=" * 80)
print("AMAZON ML CHALLENGE 2026: COMPREHENSIVE MODULE TEST SUITE")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Test config.py
# ---------------------------------------------------------------------------
print("\n--- 1. Testing config.py ---")
try:
    import config
    assert config.WORKSPACE_ROOT.exists(), "WORKSPACE_ROOT does not exist"
    assert config.TRAIN_GROUND_TRUTH.exists(), f"Ground truth not found at {config.TRAIN_GROUND_TRUTH}"
    assert config.TEST_SOURCE1.exists(), f"Test Source 1 not found at {config.TEST_SOURCE1}"
    report("config.py", "Path Resolution", True, f"Root: {config.WORKSPACE_ROOT.name}")
except Exception as e:
    report("config.py", "Path Resolution", False, str(e))

# ---------------------------------------------------------------------------
# 2. Test normalization.py
# ---------------------------------------------------------------------------
print("\n--- 2. Testing normalization.py ---")
try:
    from normalization import (
        strip_accents,
        normalize_name,
        normalize_address,
        extract_postal_code,
        extract_numeric_tokens,
        normalize_country,
        is_empty_or_nan
    )
    # Accent stripping
    assert strip_accents("Café Étoile") == "Cafe Etoile"
    report("normalization.py", "Accent Stripping", True)

    # Name normalization
    norm_n = normalize_name("The Acme & Sons, Ltd.")
    assert "acme" in norm_n and "ltd" in norm_n, f"Unexpected: {norm_n}"
    report("normalization.py", "Name Normalization", True, norm_n)

    # Indic Script preservation
    hindi = "एसएस फूड प्राइवेट लिमिटेड"
    norm_hindi = normalize_name(hindi)
    assert "एसएस" in norm_hindi and "फूड" in norm_hindi and "ltd" in norm_hindi
    report("normalization.py", "Indic Script & Suffix Standardization", True, norm_hindi)

    # Address normalization
    norm_a = normalize_address("123 Main St, Suite 400, Apt 5")
    assert "street" in norm_a and "suite" in norm_a
    report("normalization.py", "Address Expansion", True, norm_a)

    # Postal code extraction
    pin_in = extract_postal_code("New Delhi, Delhi 110001", "India")
    zip_us = extract_postal_code("Peoria, IL 61602-1234", "US")
    pin_fr = extract_postal_code("75008 Paris, France", "France")
    assert pin_in == "110001" and zip_us == "61602" and pin_fr == "75008"
    report("normalization.py", "Postal Code Extraction", True, f"IN:{pin_in}, US:{zip_us}, FR:{pin_fr}")

    # Numeric token extraction
    nums = extract_numeric_tokens("Door 42-B, Plot 104, Floor 3")
    assert "42" in nums and "104" in nums
    report("normalization.py", "Numeric Token Extraction", True, str(nums))

    # NaN safety
    assert is_empty_or_nan(float("nan")) and is_empty_or_nan("None") and is_empty_or_nan("")
    report("normalization.py", "NaN / Null Safety", True)
except Exception as e:
    report("normalization.py", "All Normalization Tests", False, str(e))

# ---------------------------------------------------------------------------
# 3. Test blocking.py
# ---------------------------------------------------------------------------
print("\n--- 3. Testing blocking.py ---")
try:
    from blocking import generate_blocking_keys, MultiKeyBlocker, CountryPartitionedBlocker

    # Blocking keys generation
    keys = generate_blocking_keys("S1-1", "Apollo Pharmacy", "12 Main Road, Chennai 600001", "India")
    assert "B1" in keys and "B2" in keys and "B4" in keys and "B6" in keys
    report("blocking.py", "Multi-Key Generation (B1-B8)", True, f"Keys generated: {list(keys.keys())}")

    # Inverted Index with uint32 array postings
    mock_cands = [
        ("S2-101", "Apollo Pharmacy", "12 Main Road", "India"),
        ("S2-102", "Apollo Hospitals", "21 Greams Road", "India"),
        ("S2-103", "MedPlus Chemist", "45 Park Street", "India")
    ]
    blocker = MultiKeyBlocker(max_candidates_per_s1=10)
    blocker.build_candidate_index_from_records(mock_cands, show_progress=False)
    assert len(blocker.index) > 0
    report("blocking.py", "Compact uint32 Inverted Index", True, f"{len(blocker.index)} buckets")

    # Candidate retrieval with IBF ranking
    retrieved = blocker.retrieve_candidates_for_record("S1-1", "Apollo Pharmacy", "12 Main Road", "India")
    assert "S2-101" in retrieved and retrieved[0] == "S2-101"
    report("blocking.py", "IBF Candidate Retrieval & Ranking", True, f"Top match: {retrieved[0]}")

    # CountryPartitionedBlocker
    cp_blocker = CountryPartitionedBlocker()
    b_india = cp_blocker.get_or_create_blocker("India")
    b_us = cp_blocker.get_or_create_blocker("US")
    assert b_india is not b_us
    cp_blocker.clear_all()
    report("blocking.py", "CountryPartitionedBlocker", True)
except Exception as e:
    report("blocking.py", "Blocking Engine Tests", False, str(e))

# ---------------------------------------------------------------------------
# 4. Test features.py
# ---------------------------------------------------------------------------
print("\n--- 4. Testing features.py ---")
try:
    from features import compute_pair_features, FEATURE_NAMES
    assert len(FEATURE_NAMES) == 24, f"Expected 24 features, got {len(FEATURE_NAMES)}"

    feat_vec = compute_pair_features(
        s1_name="Acme Supermarket",
        s1_addr="100 Main St, Austin, TX 78701",
        s1_country="US",
        cand_id="S2-999",
        cand_name="Acme Supermarket LLC",
        cand_addr="100 Main Street, Suite 2, Austin, TX 78701",
        cand_country="US"
    )
    assert len(feat_vec) == 24
    assert feat_vec[0] > 0.8  # name_ratio
    assert feat_vec[17] == 1.0  # pin_match (78701 == 78701)
    assert feat_vec[23] == 1.0  # is_source_2
    report("features.py", "24-Dimensional Feature Vector", True, f"Features length={len(feat_vec)}")
except Exception as e:
    report("features.py", "Feature Extraction Tests", False, str(e))

# ---------------------------------------------------------------------------
# 5. Test dataset.py
# ---------------------------------------------------------------------------
print("\n--- 5. Testing dataset.py ---")
try:
    from dataset import RecordCache, ChunkedDatasetBuilder, load_ground_truth_map

    # RecordCache
    sample_records = [
        ("S1-1", "Store One", "123 High St", "US"),
        ("S1-2", "Store Two", "456 Market St", "US")
    ]
    cache = RecordCache(sample_records)
    assert "S1-1" in cache.norm_names and cache.norm_names["S1-1"] == "store one"
    cache.clear()
    report("dataset.py", "RecordCache Normalized Indexing", True)

    # ChunkedDatasetBuilder
    builder = ChunkedDatasetBuilder(chunk_size=10, max_cands=5)
    assert builder.chunk_size == 10
    report("dataset.py", "ChunkedDatasetBuilder Initialization", True)
except Exception as e:
    report("dataset.py", "Dataset Pipeline Tests", False, str(e))

# ---------------------------------------------------------------------------
# 6. Test models.py
# ---------------------------------------------------------------------------
print("\n--- 6. Testing models.py ---")
try:
    from models import train_xgboost, get_feature_importances

    # Synthetic training on 24 features
    rng = np.random.RandomState(42)
    X_syn = rng.rand(100, 24).astype(np.float32)
    y_syn = (X_syn[:, 0] + X_syn[:, 9] > 1.0).astype(np.float32)

    model = train_xgboost(X_syn, y_syn, params={"n_estimators": 10, "max_depth": 3})
    preds = model.predict_proba(X_syn)[:, 1]
    assert preds.shape == (100,)
    importances = get_feature_importances(model)
    assert len(importances) == 24
    report("models.py", "XGBoost Train & Feature Importance", True, f"Top feature: {list(importances.keys())[0]}")
except Exception as e:
    report("models.py", "Model Training Tests", False, str(e))

# ---------------------------------------------------------------------------
# 7. Test decision.py
# ---------------------------------------------------------------------------
print("\n--- 7. Testing decision.py ---")
try:
    from decision import compute_entity_f05, evaluate_macro_f05, find_optimal_threshold

    # Singleton credit logic
    assert compute_entity_f05(set(), set()) == 1.0, "Singleton empty prediction must be 1.0"
    assert compute_entity_f05(set(), {"S2-1"}) == 0.0, "Singleton false positive must be 0.0"
    assert compute_entity_f05({"S2-1"}, set()) == 0.0, "Missed match must be 0.0"

    # Perfect match
    assert abs(compute_entity_f05({"S2-1"}, {"S2-1"}) - 1.0) < 1e-5
    report("decision.py", "Competition Metric Logic (Singleton & F0.5)", True)

    # Macro F0.5 evaluation
    gt = {"S1-1": {"S2-1"}, "S1-2": set()}
    pred = {"S1-1": {"S2-1"}, "S1-2": set()}
    score, p, r = evaluate_macro_f05(gt, pred)
    assert abs(score - 1.0) < 1e-5
    report("decision.py", "Macro F0.5 Evaluation", True, f"Score: {score:.2f}")

    # Threshold grid search
    pair_scores = [("S1-1", "S2-1", 0.92), ("S1-1", "S2-2", 0.30)]
    val_set = {"S1-1"}
    best_tau, best_score, _ = find_optimal_threshold(gt, pair_scores, val_set)
    assert best_tau >= 0.50 and best_score > 0
    report("decision.py", "Dynamic Threshold Search", True, f"Optimal tau: {best_tau:.2f}")
except Exception as e:
    report("decision.py", "Decision Layer Tests", False, str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"VERIFICATION SUMMARY: {passed} PASSED, {failed} FAILED")
print("=" * 80)
if failed == 0:
    print("ALL MODULES FUNCTIONING WITH 100% SUCCESS.")
    sys.exit(0)
else:
    print(f"WARNING: {failed} test(s) failed.")
    sys.exit(1)
