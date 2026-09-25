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
> **The Empirical Mindset**: We avoid making unverified assumptions. No strategy (e.g., LightGBM vs. XGBoost, specific thresholds, or blocking combinations) is declared a "winning formula" until it is empirically benchmarked against the official **Macro $F_{0.5}$** metric on our validation set.

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
        C --> D2[B2: Country + First Strong Name Token]
        C --> D3[B3: Country + Name Prefix / N-gram Key]
        C --> D4[B4: Postal Code / PIN Key]
        C --> D5[B5: Country + Significant Address Token]
        C --> D6[B6: Address Numeric + Road Token]
        C --> D7[B7: Character 3-Gram TF-IDF Cosine Retrieval]
        D1 & D2 & D3 & D4 & D5 & D6 & D7 --> E[Candidate Union & Deduplication]
        E --> E1[Measure Candidate Recall & Average Candidate Count]
        E --> F[candidate_pairs.tsv]
    end

    subgraph 3. Pairwise Feature Engineering
        F --> G[RapidFuzz String Metrics]
        F --> H[TF-IDF Cosine Similarities]
        F --> I[Structured & Numeric Overlaps]
        F --> J[PIN & Country Relational Features]
        G & H & I & J --> K[Pairwise Feature Matrix]
    end

    subgraph 4. Model Benchmarking & Hard Negative Mining
        K --> L[Entity-Level Grouped Split]
        L --> M1[LightGBM Classifier]
        L --> M2[XGBoost Classifier]
        M1 & M2 --> N[Iterative Hard Negative Mining]
        N --> O[Benchmark Winner on Validation Macro F0.5]
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

### 3.1 Data Forensics Protocol (Immediate Step 1)
Before writing production matching code, run a diagnostic script on `dataset/train/` to measure:
1. **Record Counts**: Total rows in $S_1$, $S_2$, $S_3$, and ground truth.
2. **Missing Values**: Rates of null or blank fields across sources.
3. **Cardinality Distribution**: Percentage of $S_1$ entities with 0 matches (singletons), 1 match, 2 matches, 3+ matches.
4. **Country Breakdown**: Distribution of India vs. US in train.
5. **Exact Match Prevalence**: Percentage of true pairs where `name` or `address` are character-for-character identical.
6. **Postal Code Prevalence**: Proportion of records containing extractable postal codes/PINs.
7. **Address Structural Noise**: Frequency of landmark keywords (`near`, `opp`, `behind`, `metro`), comma vs. slash separators.

---

### 3.2 Robust, Open-Set Normalization (`normalization.py`)
Normalization standardizes text representations without removing distinctive tokens:

1. **Company Name Cleaning**:
   - Lowercase and normalize Unicode characters (NFKD).
   - Strip punctuation while preserving alphanumerics and single spaces.
   - Standardize common legal entity suffixes into canonical tags:
     - `pvt ltd`, `private limited`, `pte ltd` $\rightarrow$ `__ltd__`
     - `inc`, `incorporated`, `corp`, `corporation` $\rightarrow$ `__corp__`
     - `llc`, `l.l.c.`, `llp` $\rightarrow$ `__llc__`
     - `sarl`, `s.a.r.l.`, `sa`, `sas` $\rightarrow$ `__ltd__`
     - `co`, `company`, `cie` $\rightarrow$ `__co__`
   - *Guard against over-normalization*: Retain domain keywords (`tech`, `solutions`, `bank`, `industries`).

2. **Generic Address Cleaning**:
   - Standardize common street tokens across English and international conventions:
     - `st`, `street`, `str` $\rightarrow$ `street`
     - `rd`, `road` $\rightarrow$ `road`
     - `ave`, `avenue` $\rightarrow$ `avenue`
     - `blvd`, `boulevard` $\rightarrow$ `boulevard`
     - `fl`, `flr`, `floor` $\rightarrow$ `floor`
     - `bldg`, `building` $\rightarrow$ `building`
     - `rue`, `r.` $\rightarrow$ `rue`
   - Clean whitespace, commas, hyphens, and slashes.

3. **Generic Numeric & Postal Extraction**:
   - Extract 5-digit and 6-digit postal sequences (`\b\d{5}\b`, `\b\d{6}\b`).
   - Extract building numbers and street numbers.

4. **Country Standardizer**:
   - Strips whitespace and lowercases string (`united states` $\rightarrow$ `us`, `india` $\rightarrow$ `india`, `france` $\rightarrow$ `france`).
   - Open-set: No whitelist filtering.

---

### 3.3 Multi-Key Union Blocking & Recall Benchmarking (`blocking.py`)
To prevent missing true matches while keeping candidate sets small, we evaluate a **7-Strategy Union**:

| Block ID | Key Definition | Target Match Scenario |
| :---: | :--- | :--- |
| **B1** | `country + "_" + clean_name` | Exact or near-exact business names |
| **B2** | `country + "_" + first_strong_name_token` | Legal suffix changes, trailing word additions |
| **B3** | `country + "_" + clean_name[:4]` | Typos in second/third words, slight renames |
| **B4** | `extracted_postal_code` | Completely different spelling/typo in same building/zone |
| **B5** | `country + "_" + rare_address_token` | Distinctive locality / tech park / street names |
| **B6** | `country + "_" + street_number + "_" + street_token` | Exact building address with modified company name |
| **B7** | Character 3-Gram TF-IDF Cosine (Top-15 neighbors) | Severe misspellings, transliterations |

$$\text{Candidate Set}(S_1) = \bigcup_{i=1}^{7} \text{Block}_i(S_1)$$

#### Blocking Diagnostics to Measure on Validation Split:
- **Candidate Recall**: $\frac{\text{True matches present in candidate set}}{\text{Total true matches}}$ (Target: $> 98\%$)
- **Average Candidate Count**: $\frac{\text{Total candidates}}{N_{S_1}}$ (Target: $\le 40$ candidates per $S_1$)
- **Reduction Ratio**: $1 - \frac{\text{Generated pairs}}{N_{S_1} \times (N_{S_2} + N_{S_3})}$ (Target: $> 99.9\%$)

---

### 3.4 Pairwise Feature Matrix (`features.py`)
Features are computed across multiple distinct similarity dimensions:

#### A. Name Similarities (RapidFuzz & TF-IDF)
1. `name_ratio`: Full Levenshtein similarity ratio.
2. `name_partial_ratio`: Substring alignment score.
3. `name_token_sort_ratio`: Word-order invariant similarity.
4. `name_token_set_ratio`: Set intersection similarity (handles omitted/extra words).
5. `name_w_ratio`: Weighted combination of partial and token methods.
6. `name_jaro_winkler`: Prefix-weighted typo similarity.
7. `name_char_tfidf_cosine`: Character 3-gram TF-IDF cosine similarity.
8. `name_word_jaccard`: Exact word token Jaccard similarity.
9. `name_len_diff_ratio`: Normalized length difference.

#### B. Address Similarities
10. `addr_ratio`: Full string similarity.
11. `addr_partial_ratio`: Substring match score (detects street within long address).
12. `addr_token_sort_ratio`: Component reordering invariant score.
13. `addr_token_set_ratio`: Shared token core score.
14. `addr_char_tfidf_cosine`: Character n-gram TF-IDF cosine similarity.
15. `addr_word_jaccard`: Word token Jaccard similarity.
16. `addr_len_diff_ratio`: Normalized length difference.

#### C. Numeric & Structured Features (Empirical / Non-Hardcoded)
17. `pin_match`: Binary flag (1 if both have PIN and PINs match, 0 otherwise).
18. `pin_mismatch`: Binary flag (1 if both have PIN and PINs differ, 0 otherwise).
19. `pin_missing_either`: Binary flag (1 if either or both lack a PIN, 0 otherwise).
20. `pin_both_present`: Binary flag (1 if both have an extractable PIN, 0 otherwise).
21. `numeric_token_jaccard`: Jaccard similarity of extracted numbers (building/floor/door).
22. `has_common_number`: Binary flag (1 if at least one multi-digit number matches).
23. `country_exact`: Binary flag ($1.0$ if $S_1\text{.country} == S_{cand}\text{.country}$ else $0.0$).
24. `is_source_2`: Indicator for candidate origin ($S_2$).
25. `is_source_3`: Indicator for candidate origin ($S_3$).

---

### 3.5 Validation Methodology & Hard Negative Mining (`training.py`)

#### 1. Entity-Level Grouped Split (Zero Data Leakage)
- Split by **`source1_entity_id`** (80% train, 20% validation).
- An $S_1$ entity and all its candidate pairs exist **exclusively in train or exclusively in validation**.

#### 2. Iterative Hard Negative Mining Protocol
- **Initial Training Set**:
  - Positive pairs: All true matches from `train_ground_truth.tsv` in the training split.
  - Initial negative pairs: Sampled from the candidate generation stage (pairs blocked together but not in ground truth).
- **Hard Negative Mining Cycle**:
  1. Train initial model on positive + candidate negative pairs.
  2. Run inference over the training split's full candidate pool.
  3. Identify false positives where predicted probability $P > 0.40$ but label is $0$.
  4. Augment training set with these mined hard negatives.
  5. Retrain model with balanced sample weights or controlled positive-to-negative ratio ($1:5$).

#### 3. Head-to-Head Model Benchmark: LightGBM vs. XGBoost
Train both classifiers on the exact same folds and feature sets:
- **LightGBM**: Histogram-based, leaf-wise growth, fast iterations.
- **XGBoost**: Exact / approx split, depth-wise growth, robust regularization.
- **Decision Criterion**: Select the model that achieves the higher **Macro $F_{0.5}$** on the validation set.

---

### 3.6 Precision-Heavy Decision Engine & Threshold Search (`matching.py`)

#### 1. Empirical Threshold Optimization
- Run a fine-grained grid search: $\tau \in [0.50, 0.99]$ with step $0.01$.
- At each $\tau$, calculate the exact competition metric:
  $$\text{Macro } F_{0.5} = \frac{1}{N_{S_1}} \sum_{i=1}^{N_{S_1}} F_{0.5}(S_{1, i}, \hat{Y}_i(\tau))$$
- Identify the empirical peak $\tau^*$ that maximizes Macro $F_{0.5}$.

#### 2. Singleton & Score-Gap Diagnostics
For each $S_1$ entity with candidate probabilities $\{p_1, p_2, \dots, p_k\}$:
- Sort probabilities in descending order: $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(k)}$.
- Compute diagnostic features:
  - `candidate_count`: $k$
  - `max_score`: $p_{(1)}$
  - `second_score`: $p_{(2)}$ (or $0.0$ if $k < 2$)
  - `score_gap`: $p_{(1)} - p_{(2)}$
- **Singleton Rule**: If $k == 0$ or $p_{(1)} < \tau^*$, emit an empty string `""` (earning score 1.0 if true singleton).
- **Multi-Match Rule**: Emit all candidates where $p_j \ge \tau^*$.
- **Ambiguity Guard**: If a secondary candidate $p_{(2)} \ge \tau^*$ but has a large score gap ($p_{(1)} - p_{(2)} > \Delta_{guard}$), evaluate whether secondary filtering improves validation precision.

---

## 4. Submission Package Architecture

The solution generates the required zip archive matching competition rules:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv       # S1 entity matches (Uploaded to Leaderboard)
│   └── candidate_pairs.tsv        # Evaluated candidate set from blocking
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py          # Paths, thresholds, hyperparams
│       │   ├── forensics.py       # Dataset analysis & statistics
│       │   ├── normalization.py   # Multi-language text & address cleanups
│       │   ├── blocking.py        # 7-key candidate generation & recall tracker
│       │   ├── features.py        # RapidFuzz & TF-IDF feature extractor
│       │   ├── dataset.py         # Entity-grouped splits & hard negative miner
│       │   ├── models.py          # LightGBM & XGBoost training/benchmarking
│       │   ├── decision.py        # Macro F0.5 grid search & inference engine
│       │   └── run_pipeline.py    # Master end-to-end execution script
│       ├── README.md              # Complete reproduction guide
│       └── requirements.txt       # Pinned dependencies
└── Documentation_template.md      # Completed competition methodology write-up
```

---

## 5. Execution Roadmap & Milestones

| Step | Phase | Key Milestone | Verification Gate |
| :---: | :--- | :--- | :--- |
| **01** | **Dataset Forensics** | Extract `student_resource.zip` and run diagnostic inspection on all training tables. | Cardinality, singleton %, noise rates logged. |
| **02** | **Normalization** | Implement and test string/address cleansers on US, India, and France samples. | Normalized strings inspected for preservation. |
| **03** | **Multi-Key Blocking** | Benchmark 7 blocking keys and measure Candidate Recall & Candidate Count. | Candidate Recall $> 98\%$, Avg Cand $\le 40$. |
| **04** | **Feature Engineering** | Build C++ RapidFuzz + TF-IDF matrix computation. | Feature extraction throughput verified. |
| **05** | **Benchmark Models** | Train LightGBM vs. XGBoost on identical entity-grouped splits with hard negatives. | Best model selected via Validation Macro $F_{0.5}$. |
| **06** | **Decision Tuning** | Grid search $\tau \in [0.50, 0.99]$ on validation split; verify singleton handling. | Optimal $\tau^*$ identified; gap analysis logged. |
| **07** | **Inference & Checks**| Generate test `matching_results.tsv` and `candidate_pairs.tsv`. Run `utils/validate_submission.py`. | Validator exits with code 0 (`PASS`). |
| **08** | **Zip Packaging** | Populate `Documentation_template.md` and assemble submission archive. | Structure matches competition requirements. |
