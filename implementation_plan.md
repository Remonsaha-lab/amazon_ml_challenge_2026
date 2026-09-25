# Amazon ML Challenge 2026: Business Entity Resolution
## Empirical Implementation Plan & Master System Architecture

---

### Executive Overview & Empirical Philosophy

The **Amazon ML Challenge 2026** is an **Entity Resolution (Record Linkage)** competition across three heterogeneous, noisy data sources:
- **Source 1 ($S_1$)**: Deduplicated, clean reference source.
- **Source 2 ($S_2$)**: Noisy business listings (abbreviations, typos, missing elements).
- **Source 3 ($S_3$)**: Noisy business listings (transliterations, landmarks, alternate forms).

The core mission: **For every single entity in $S_1$, identify all corresponding matching records in $S_2$ and $S_3$ without any shared unique identifier.**

> [!IMPORTANT]
> **The Empirical Mindset**: We avoid making unverified assumptions. Every strategy (blocking combinations, feature sets, model choices, and decision thresholds) is empirically benchmarked against the official **Macro $F_{0.5}$** metric on our validation set.

```text
Hypothesis ──► Implement ──► Measure on Validation ──► Retain if Metric Improves ──► Discard / Refine if Not
```

---

## 1. Problem Formulation & Competition Rules Analysis

### 1.1 The Evaluation Metric: Macro $F_{0.5}$
The evaluation uses the macro-averaged $F_{0.5}$ score across all $S_1$ entities:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}} = \frac{(1 + 0.5^2) \times P \times R}{0.5^2 \times P + R}$$

#### Critical Metric Mechanics:
1. **Precision is weighted 2× over Recall**: Falsely merging two distinct businesses destroys the score much more severely than a missed link ($0.25 P + R$).
2. **Per-Entity Macro Average**: $F_{0.5}$ is computed individually for every $S_1$ entity in the evaluation set, then averaged.
3. **The Singleton Rule ($1.0$ vs $0.0$)**:
   - If an $S_1$ entity has **no true matches** (a singleton) and the model predicts an empty list, it receives a score of **$1.0$**.
   - If the model predicts any false match for that singleton, its score plunges to **$0.0$**.
   - Correctly identifying "no match" is a first-class objective, not an afterthought.

### 1.2 Open-Set Country Generalization (The "France" Trap)
- **Training Data**: US and India.
- **Test Data**: US, India, and **France**.
- **Rule**: Country **must not** be hardcoded, filtered, or one-hot encoded with fixed categories.
- **Generic Parsing**: The parser extracts generic patterns (postal code digits, street tokens, numeric addresses) without external geographic lookups. No hand-engineered country lookups (e.g., mapping `75001 -> Paris`) are used, ensuring 100% compliance with fair-play rules.

### 1.3 Key Competition Constraints
- **Strictly Zero External Lookups**: No Google Maps, Nominatim, OpenStreetMap, company registries, or internet lookups.
- **Model License & Scale**: Final model must be under **MIT / Apache 2.0 License** and $\le 8$ billion parameters. (LightGBM / XGBoost with RapidFuzz satisfies this completely).
- **Dual TSV Deliverables**:
  1. `output/matching_results.tsv`: `source1_entity_id \t matched_entity_ids` (Final matches; uploaded to leaderboard).
  2. `output/candidate_pairs.tsv`: `source1_entity_id \t candidate_entity_ids` (The exact candidate set passed to the final ML model).
- **Validation Script**: `python3 utils/validate_submission.py --matching ... --candidate ... --test-dir ...` must exit with code 0 (`PASS`).

---

## 2. Refined End-to-End System Pipeline

```mermaid
graph TD
    subgraph 1. Data Ingestion & Forensics
        A[Raw TSV: S1, S2, S3, Ground Truth] --> B[Dataset Forensics & Diagnostics]
        B --> C[Robust Normalization Engine]
    end

    subgraph 2. Multi-Key Union Blocking (High Recall)
        C --> D1[B1: Country + Exact Clean Name]
        C --> D2[B2: Country + All Significant Name Tokens]
        C --> D3[B3: Country + Name Prefix Key]
        C --> D4[B4: Postal Code / PIN Key]
        C --> D5[B5: Large Numbers / Identifiers >= 4 digits]
        C --> D6[B6: Door Number + Locality Token]
        C --> D7[B7: Distinctive Locality Tokens >= 6 chars]
        D1 & D2 & D3 & D4 & D5 & D6 & D7 --> E[Inverted Index with IBF Ranking]
        E --> E1[Candidate Recall: 95.48%, Avg Cands: 49.4]
        E --> F[candidate_pairs.tsv]
    end

    subgraph 3. Pairwise Feature Engineering
        F --> G[RapidFuzz String Metrics: Ratio, Partial, Token Sort, Token Set, JW]
        F --> H[Address Overlaps & Empty Address Indicators]
        F --> I[Numeric Door / Plot Number Jaccard]
        F --> J[Empirical PIN Match / Mismatch / Missing Flags]
        G & H & I & J --> K[24-Dimensional Pairwise Feature Matrix]
    end

    subgraph 4. Model Benchmarking & Hard Negative Mining
        K --> L[Entity-Level Grouped Split]
        L --> M1[LightGBM Classifier: 0.9581 Macro F0.5]
        L --> M2[XGBoost Classifier: 0.9619 Macro F0.5 - Winner]
        M1 & M2 --> N[Iterative Hard Negative Mining: 1 Pos to 4 Negs]
        N --> O[Winning Model: XGBoost with tau = 0.84]
    end

    subgraph 5. Decision Engine & Verification
        O --> P[Fine-Grained Grid Search: Threshold tau in 0.50 to 0.99]
        P --> Q[Diagnostic Filtering: Candidate Count & Score Gap]
        Q --> R[matching_results.tsv]
        F & R --> S[utils/validate_submission.py]
        S --> T[team_submission.zip]
    end
```

---

## 3. Systematic Component Specifications

### 3.1 Data Forensics Summary (Ground-Truth Findings)
Inspecting the full dataset across 12,527,040 training records and 11,702,133 test records revealed:
1. **Total $S_1$ Entities**: 2,206,821
2. **True Match Pairs**: 7,638,365 pairs ($\approx 3.46$ matches per $S_1$)
3. **Cardinality Breakdown**:
   - Singletons (0 matches): 123,247 (5.58%)
   - 1 match: 119,157 (5.40%)
   - 2 matches: 375,212 (17.00%)
   - **3+ matches**: **1,589,205 (72.01%)**
   - **Both $S_2$ and $S_3$ matched simultaneously**: **80.48%**!
4. **Country Breakdown**:
   - Train $S_1$: US (60.0%), India (40.0%)
   - Test $S_1$: India (46.8%), US (38.3%), France (15.0% - 259k entities)

---

### 3.2 Robust Normalization (`normalization.py`)
- **Selective Latin Accent Stripping**: Strips combining accents (`É` $\rightarrow$ `E`, `ó` $\rightarrow$ `o`) only in Latin ranges, preserving Indic scripts (Devanagari, Tamil, Kannada) 100% intact.
- **Unicode Category Symbol Stripping**: Preserves Letters (L), Indic Vowels/Marks (M), Numbers (N), and whitespace, while safely stripping Punctuation (P) and Symbols (S).
- **Multilingual Legal Entity Standardization**: Normalizes legal designations across English, Hindi, and Tamil: `pvt ltd` / `private limited` / `प्रा. लि.` / `பிரைவேட் லிமிடெட்` $\rightarrow$ `ltd`; `LLC` / `LLP` / `எல்எல்பி` $\rightarrow$ `llc`.
- **Address & State Normalization**: Expands common state codes (`NY`, `CA`, `IL`, `TN`, `UP`, `DL`, `KA`) and standardizes street terms (`street`, `road`, `avenue`, `floor`, `rue`).
- **Alphanumeric & Postal Code Parsing**: Universal extraction of door/plot numbers (`wz-187c`, `af-684`) and postal codes across US (5-digit), India (6-digit), and France (5-digit).

---

### 3.3 Multi-Key Blocking with IBF Candidate Ranking (`blocking.py`)
To prevent candidate crowding from generic words while achieving high recall, candidates are scored using **Inverse Bucket Frequency (IBF)**:
$$\text{Score}(\text{cand}) = \sum_{k \in \text{Keys}(S_1) \cap \text{Keys}(\text{cand})} \frac{\text{Weight}(\text{strategy})}{\log_2(2 + \text{BucketSize}(k))}$$
- **Theoretical Union Recall**: **99.08%**
- **Verified Benchmark Recall**: **95.48%** (17,417 / 18,242 links recovered)
- **Average Candidates per $S_1$**: **49.4** (adhering to $\le 50$ cap)
- **Reduction Ratio**: **99.9706%**

---

### 3.4 Pairwise Feature Matrix (`features.py`)
A 24-dimensional feature vector extracted at $>1,700$ records/sec using C++ `RapidFuzz`:
- **Name Features (0-8)**: `name_ratio`, `name_partial_ratio`, `name_token_sort_ratio`, `name_token_set_ratio`, `name_w_ratio`, `name_jaro_winkler`, `name_word_jaccard`, `name_len_diff_ratio`, `name_exact_clean`.
- **Address Features (9-15)**: `addr_ratio`, `addr_partial_ratio`, `addr_token_sort_ratio`, `addr_token_set_ratio`, `addr_word_jaccard`, `addr_len_diff_ratio`, `addr_is_empty_either`.
- **Geographic & PIN Features (16-20)**: `country_exact`, `pin_match`, `pin_mismatch`, `pin_missing_either`, `pin_both_present`.
- **Numeric & Source Overlaps (21-23)**: `numeric_token_jaccard`, `has_common_number`, `is_source_2`.

---

### 3.5 Model Training & Macro $F_{0.5}$ Benchmarking (`models.py`, `decision.py`)
- **Entity-Grouped Split**: Strictly split on `source1_entity_id` (80% train, 20% val) with zero data leakage.
- **Hard Negative Mining**: Mined 4 authentic hard negatives per positive pair directly from the candidate pool.
- **Head-to-Head Results**:
  - LightGBM: Optimal $\tau = 0.84$ $\rightarrow$ Validation Macro $F_{0.5} = \mathbf{0.9581}$
  - **XGBoost: Optimal $\tau = 0.84$ $\rightarrow$ Validation Macro $F_{0.5} = \mathbf{0.9619}$ (Winner)**
- **Top Feature Contributions**:
  - `addr_token_set_ratio`: **51.90%**
  - `addr_word_jaccard`: **18.98%**
  - `addr_is_empty_either`: **8.27%**
  - `addr_token_sort_ratio`: **7.30%**
  - `name_w_ratio`: **2.98%**

---

## 4. Submission Package Architecture

The solution generates the required zip archive matching competition rules:

```text
AWS_ml_challenge/
├── code/
│   └── business_entity_resolution/
│       └── src/
│           ├── __init__.py           # Package marker
│           ├── config.py             # Global constants, paths, thresholds, hyperparameters
│           ├── normalization.py      # Unicode accent stripper, legal canonicalizer, regex parsers
│           ├── blocking.py           # 7-Key inverted index with IBF candidate ranking
│           ├── features.py           # RapidFuzz 24-dimensional pairwise feature vector extractor
│           ├── dataset.py            # RecordCache, grouped splits, and hard negative miner
│           ├── models.py             # XGBoost & LightGBM training wrappers
│           ├── decision.py           # Macro F0.5 evaluation metric & grid-search threshold optimizer
│           ├── forensics.py          # Data auditing, cardinality, and distribution analyzer
│           └── run_pipeline.py       # Master end-to-end training, streaming inference & submission writer
├── student_resource/                 # Competition provided resources
│   ├── dataset/
│   │   ├── train/                    # 12.5M records (train_source1/2/3.tsv, train_ground_truth.tsv)
│   │   └── test/                     # 11.7M records (test_source1/2/3.tsv)
│   ├── utils/
│   │   └── validate_submission.py    # Official submission format validation script
│   └── Documentation_template.md     # Competition write-up template
├── output/                           # Target directory for generated submissions
│   ├── matching_results.tsv          # Final matched pairs (uploaded to leaderboard)
│   └── candidate_pairs.tsv           # Candidate pairs output by blocking stage
├── benchmark_blocking.py             # Benchmark measuring blocking recall on ground truth
├── benchmark_models.py               # Head-to-head XGBoost vs LightGBM benchmark
├── test_normalization.py             # Normalization unit tests (Devanagari, Tamil, French, US)
├── diagnose_blocking_recall.py       # Diagnostic script analyzing theoretical recall ceilings
├── debug_missed_blocks.py            # In-depth inspector for false negatives in blocking
├── test_candidate_ranking.py         # Diagnostic for IBF candidate ranking scoring
├── test_indic.py                     # Indic unicode preservation probe
└── sample_pairs.py                   # Ground-truth pair inspector for manual qualitative analysis

```

---

## 5. Execution Roadmap & Milestones

| Step | Phase | Key Milestone | Verification Gate |
| :---: | :--- | :--- | :--- |
| **01** | **Dataset Forensics** | Extract `student_resource.zip` and run diagnostic inspection on all training tables. | Cardinality, singleton %, noise rates logged. ✅ |
| **02** | **Normalization** | Implement and test string/address cleansers on US, India, and France samples. | Normalized strings inspected for preservation. ✅ |
| **03** | **Multi-Key Blocking** | Benchmark 7 blocking keys and measure Candidate Recall & Candidate Count. | Candidate Recall: **95.48%**, Avg Cand: **49.4**. ✅ |
| **04** | **Feature Engineering** | Build C++ RapidFuzz 24-dim matrix computation. | Extraction speed: **1,714 records/sec**. ✅ |
| **05** | **Benchmark Models** | Train LightGBM vs. XGBoost on identical entity-grouped splits with hard negatives. | **XGBoost selected**: Validation Macro $F_{0.5} = \mathbf{0.9619}$. ✅ |
| **06** | **Decision Tuning** | Grid search $\tau \in [0.50, 0.99]$ on validation split; verify singleton handling. | Optimal threshold identified: **$\tau^* = 0.84$**. ✅ |
| **07** | **Inference & Checks**| Generate test `matching_results.tsv` and `candidate_pairs.tsv`. Run `utils/validate_submission.py`. | Validator exits with code 0 (`PASS`). |
| **08** | **Zip Packaging** | Populate `Documentation_template.md` and assemble submission archive. | Structure matches competition requirements. |

---

## 6. Complete Verification, Testing, Training & Evaluation Command Handbook

Run each command directly in your terminal from the workspace root (`c:\Users\remon\OneDrive\Documents\AWS_ml_challenge`):

### 6.1 Environment & Dependency Verification
Verifies that all Python 3.12 dependencies are correctly installed:
```powershell
python check_env.py
```
*Expected Output: `pandas`, `numpy`, `scipy`, `sklearn`, `lightgbm`, `xgboost`, `rapidfuzz`, and `tqdm` reported as installed.*

---

### 6.2 Step 1: Dataset Forensics & Ground-Truth Statistics
Runs full forensic inspection across training and test TSVs:
```powershell
python forensics.py
```
*Expected Output: Line counts across ~12.5M train / ~11.7M test records, singleton rate (5.58%), cardinality (72% 3+ matches), and US/India/France distributions.*

---

### 6.3 Step 2: Normalization Unit Tests & Verification
Verifies Latin accent stripping, Indic script preservation (Devanagari, Tamil), alphanumeric door extraction, and postal codes:
```powershell
python test_normalization.py
```
*Expected Output: `ALL NORMALIZATION ASSERTIONS PASSED! (Verification Gate 02 Complete)`*

---

### 6.4 Step 3: Candidate Generation / Blocking Benchmark
Evaluates Multi-Key Blocking over 18,242 genuine ground-truth links across a 168k candidate pool:
```powershell
python benchmark_blocking.py
```
*Expected Output: Candidate Recall $\ge 95\%$, Average Candidates/S1 $\le 50$, Reduction Ratio $> 99.9\%$, Verification Gate 03 PASSED.*

---

### 6.5 Blocking Diagnostics & Error Analysis (Optional Deep Dive)
Inspects the theoretical recall ceiling and analyzes individual missed pairs:
```powershell
# 1. Theoretical recall ceiling across all 7 keys without bucket caps:
python diagnose_blocking_recall.py

# 2. Inspect missed pairs and key overlap:
python debug_missed_blocks.py
```

---

### 6.6 Step 5: Head-to-Head Model Benchmarking & Threshold Search
Trains LightGBM and XGBoost on identical entity-grouped splits with hard negatives and grid-searches $\tau \in [0.50, 0.99]$ on the official Macro $F_{0.5}$ metric:
```powershell
python benchmark_models.py
```
*Expected Output: Model comparison table, optimal threshold ($\tau = 0.84$), validation Macro $F_{0.5}$ scores (XGBoost: 0.9619), and top 8 feature importances.*

---

### 6.7 Step 6: Master End-to-End Pipeline Execution
Trains the production matcher, indexes the test candidate pool, streams test entities, and generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```powershell
python code/business_entity_resolution/run_pipeline.py
```
*Expected Output: Progress bar indexing test pool, streaming candidate scoring, and writing output files.*

---

### 6.8 Step 7: Official Submission Format Validation
Runs the official challenge validator against the generated output files:
```powershell
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
```
*Expected Output: `PASS — no blocking issues found. Safe to submit.` (Exit Code 0).*

---

### 6.9 Step 8: Git Commit & Push (Excluding Datasets & Videos)
Stages and pushes only clean source code, markdown documentation, and configuration files to GitHub:
```powershell
git add .
git commit -m "Implement end-to-end Entity Resolution pipeline with 95.5% candidate recall and 0.962 Macro F0.5 XGBoost matcher"
git push -u origin main
```
