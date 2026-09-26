# Amazon ML Challenge 2026: Business Entity Resolution
## Scalable Country-Partitioned XGBoost Matching Architecture

This package contains the complete, reproducible solution for the **Amazon ML Challenge 2026 Business Entity Resolution** task.

---

### System Architecture Overview

1. **Script-Preserving Text Normalization (`normalization.py`)**:
   - Strips Latin diacritics while preserving Devanagari and Indic scripts.
   - Robust `is_empty_or_nan()` checks preventing literal `"nan"` or `"none"` leakage.
   - Regex-based postal code extraction (US 5-digit ZIPs, India 6-digit PINs, France 5-digit postal codes).
   - Address abbreviation expansion (St -> Street, Rd -> Road, Av -> Avenue, etc.).

2. **Low-Memory Inverted Index Blocking (`blocking.py`)**:
   - Employs 8 complementary blocking strategies:
     - `B1`: Exact Normalized Name
     - `B2`: Significant Name Tokens
     - `B3`: 4-Character Name Prefix
     - `B4`: Postal Code / PIN
     - `B5`: Numeric Identifiers (>= 4 digits)
     - `B6`: Door / Building Number + Locality Token
     - `B7`: Distinctive Locality Tokens
     - `B8`: Character 3-Gram Subwords (typo/transliteration resilience)
   - Uses compact uint32 postings (`array('I')`) consuming only 4 bytes per posting instead of 200+ bytes for Python sets.
   - Bucket-size ingestion caps (`max_bucket_size=2500`) prevent stopword explosion.
   - Inverse Bucket Frequency (IBF) candidate ranking prioritizes rare, high-information matches.

3. **Pairwise Feature Engineering (`features.py`)**:
   - 24-dimensional feature vector per candidate pair using C++ accelerated RapidFuzz metrics:
     - Name similarity: Ratio, Partial Ratio, Token Sort, Token Set, WRatio, Jaro-Winkler, Word Jaccard.
     - Address similarity: Ratio, Partial Ratio, Token Sort, Token Set, Word Jaccard.
     - Geographic & PIN indicators: exact country match, PIN match, PIN mismatch, PIN presence flags.
     - Numeric token overlap: Door/plot number Jaccard similarity and common number indicators.

4. **Chunked Training & Dynamic Threshold Optimization (`dataset.py`, `models.py`, `decision.py`)**:
   - `ChunkedDatasetBuilder`: processes S1 entities in memory-safe chunks (e.g., 10,000 entities), extracting features and freeing memory between chunks.
   - Hard negative mining: extracts 1:4 hard negatives based on top-ranked false positive candidates from blocking.
   - Entity-grouped 80/20 train/validation split.
   - XGBoost classifier with early stopping.
   - Dynamic threshold grid search optimizing decision threshold $\tau^*$ directly on official competition Macro $F_{0.5}$ metric (optimal $\tau^* \approx 0.85$).

5. **Streaming Country-Partitioned Test Inference (`run_pipeline.py`)**:
   - Streams candidates by country (France -> US -> India) in isolated passes.
   - Releases candidate indices between passes, keeping peak RAM $< 1.5$ GB across all 10M candidates.
   - Streams results directly to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

---

### Installation & Prerequisites

Python 3.8+ is required. Install dependencies via:

```bash
pip install -r requirements.txt
```

---

### How to Run

From the project root:

```bash
# Full production run across all 1.73M test records
python code/business_entity_resolution/src/run_pipeline.py --full

# Fast verification run (processes first 2,000 test records)
python code/business_entity_resolution/src/run_pipeline.py --test-sample 2000

# Custom training cohort and chunk size
python code/business_entity_resolution/src/run_pipeline.py --full --train-cohort 30000 --chunk-size 10000 --max-cands 50

python code/business_entity_resolution/src/run_pipeline.py --full --train-cohort 50000 --chunk-size 10000 --max-cands 50

```

---

### Output Deliverables

The pipeline writes two official TSV deliverables to `output/`:
- `output/matching_results.tsv` (Leaderboard submission: `source1_entity_id \t matched_entity_ids`)
- `output/candidate_pairs.tsv` (Blocking candidate sets: `source1_entity_id \t candidate_entity_ids`)
