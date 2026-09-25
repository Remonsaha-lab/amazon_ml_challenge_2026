"""
Configuration module for Amazon ML Challenge 2026: Business Entity Resolution.
Supports running from workspace root or inside the package directory.
"""

import os
from pathlib import Path

# Resolve base directories
CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent.parent

# Data directories (prefers student_resource/dataset if present, fallback to dataset/)
if (WORKSPACE_ROOT / "student_resource" / "dataset").exists():
    DATA_DIR = WORKSPACE_ROOT / "student_resource" / "dataset"
elif (WORKSPACE_ROOT / "dataset").exists():
    DATA_DIR = WORKSPACE_ROOT / "dataset"
else:
    DATA_DIR = CURRENT_DIR / "dataset"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Output directories
OUTPUT_DIR = WORKSPACE_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Training files
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# Test files
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Pipeline Parameters
RANDOM_SEED = 42
VALIDATION_SPLIT_RATIO = 0.20  # 20% holdout grouped by source1_entity_id

# Blocking Limits
MAX_CANDIDATES_PER_S1 = 50     # Safeguard against candidate blowup on generic names
TFIDF_TOP_K = 15               # Top K nearest neighbors via char 3-gram TF-IDF

# Hard Negative Mining Parameters
HARD_NEGATIVE_RATIO = 4        # 1 positive : 4 hard negatives
HARD_NEG_MIN_PROB = 0.35       # Mining threshold for false positives

# Decision Layer
THRESHOLD_GRID_START = 0.50
THRESHOLD_GRID_END = 0.99
THRESHOLD_GRID_STEP = 0.01
