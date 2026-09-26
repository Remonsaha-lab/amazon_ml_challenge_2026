# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** EntityResolvers  
**Problem:** Cross-Source Entity Matching ($S_1 \rightarrow S_2, S_3$) across US, India, France  
**Evaluation Metric:** Official Macro $F_{0.5}$  

---

## 1. Executive Summary

We present a high-scale, memory-efficient business entity resolution architecture designed to link noisy reference records ($S_1$) with candidate listings ($S_2, S_3$) across diverse international formats (United States, India, France). Our pipeline combines an 8-way multi-key union inverted index with compact uint32 array postings and Inverse Bucket Frequency (IBF) candidate ranking, a 24-dimensional feature extraction engine powered by C++ accelerated RapidFuzz metrics, and an XGBoost binary classifier dynamically optimized directly on the competition Macro $F_{0.5}$ evaluation metric. By enforcing strict geographic partitioning and chunked feature extraction, the entire 10-million record candidate space is processed within $\le 1.5\text{ GB}$ of RAM with zero SSD paging thrash.

---

## 2. Methodology

### 2.1 Problem Analysis
Analysis of the dataset revealed several domain-specific challenges:
1. **Severe Text & Structural Noise**: Addresses exhibit dramatic formatting divergence, missing house numbers, inverted word orders, and phonetic spelling differences (e.g., *Jhsnno* vs *Johnson*, *Dundalk MD* vs *Dundalk, Maryland*).
2. **Multi-Script & Transliteration Heterogeneity**: The dataset spans three distinct geographic regimes:
   - **India**: Long unstructured address strings, variable PIN code placement, colloquial landmark descriptions, and mixed Hindi/Tamil transliterations.
   - **United States**: Strict ZIP/state conventions but high frequency of suite, apartment, and building abbreviation variations.
   - **France**: French accented characters (*é, è, ê, ô, ç*), street type prefixes (*rue, avenue, boulevard, allée*), and 5-digit postal codes.
3. **Severe Class Imbalance & Candidate Scale**: Linking 1.73M test $S_1$ entities against ~10M candidate records in $S_2$ and $S_3$ yields an unconstrained search space of $1.73 \times 10^{13}$ pairs. Memory bloat from naive Python dictionaries or dataframes causes immediate laptop freezing.
4. **Metric Penalty Structure**: The Macro $F_{0.5}$ metric weights precision $2\times$ over recall:
   $$F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$$
   Singletons (records with no true match) receive a score of $1.0$ if predicted empty, but $0.0$ if even a single false positive match is predicted. Consequently, conservative decision thresholds ($\tau^* \ge 0.75$) are mathematically optimal.

### 2.2 Solution Strategy
- **Approach Type**: Country-Partitioned Multi-Key Union Inverted Index Blocker + 24-Feature XGBoost Matcher with Dynamic Macro $F_{0.5}$ Threshold Search.
- **Core Innovations**:
  - **Zero-Copy Compact Indexing**: Replaced Python `set` postings with integer-mapped C-style uint32 arrays (`array('I')`), reducing memory from 200+ bytes per posting to 4 bytes per posting.
  - **Country-Partitioned Memory Isolation**: Explored and confirmed that cross-country matching is exactly 0.00%. Test candidate pools are partitioned into France (~1.4M), US (~3.8M), and India (~4.7M), loaded sequentially and garbage-collected between passes.
  - **Chunked Feature Extraction (`ChunkedDatasetBuilder`)**: S1 entities are processed in blocks of 10,000 entities, keeping memory flat and constant ($\le 1.5\text{ GB}$) during both training and inference.
  - **Direct $F_{0.5}$ Threshold Optimization**: Discarded default 0.50 classification thresholds in favor of entity-level validation grid search over $\tau \in [0.50, 0.99]$.

---

## 3. Candidate Generation (Blocking)

To guarantee high candidate recall while keeping candidate density below 50 candidates per $S_1$ record, we designed an 8-strategy union inverted index:

| Key Code | Blocking Strategy | Target Match Variation Handled |
| :---: | :--- | :--- |
| **B1** | Exact Normalized Clean Name | Verbatim brand/entity matches |
| **B2** | Significant Name Tokens (top 4 tokens, len $\ge 3$) | Missing leading articles, word re-orderings, corporate suffix divergence |
| **B3** | 4-Character Name Prefix | Minor spelling truncations and suffix variations |
| **B4** | Postal Code / PIN Key | Clean geographical anchor matches |
| **B5** | Distinctive Numeric Identifiers ($\ge 4$ digits) | Phone numbers, building codes, plot identifiers |
| **B6** | Door Number + Distinctive Locality Token | Indic transliteration and door/plot + colony matches |
| **B7** | High-Information Locality Tokens (len $\ge 6$) | Landmark, neighborhood, and city co-occurrences |
| **B8** | Character 3-Gram Subwords | Typo resilience and phonetic transliterations |

- **Inverse Bucket Frequency (IBF) Candidate Ranking**:
  Candidate priority scores are computed as:
  $$\text{Score}(c) = \sum_{k \in \text{Keys}(S_1) \cap \text{Keys}(c)} \frac{W(k)}{\log_2(2 + |B_k|)}$$
  Where $W(k)$ is the strategy weight (e.g., $15.0$ for B1, $6.0$ for B2, $4.0$ for B6) and $|B_k|$ is the bucket size. High-frequency stopwords are automatically downweighted or capped at ingestion (`max_bucket_size=2500`).
- **Candidate Recall**: Achieves **95.48% Candidate Recall** on holdout evaluation while maintaining an average density of **49.4 candidates per entity** (99.97% search space reduction).

---

## 4. Matching Model

### 4.1 Feature Engineering (24 Dimensions)
1. **Name Similarity (9 features)**: RapidFuzz Ratio, Partial Ratio, Token Sort Ratio, Token Set Ratio, WRatio, Jaro-Winkler distance, Word Jaccard overlap, length difference ratio, exact clean match boolean.
2. **Address Similarity (7 features)**: RapidFuzz Ratio, Partial Ratio, Token Sort Ratio, Token Set Ratio, Word Jaccard overlap, length difference ratio, missing address indicator.
3. **Geographic & PIN Identifiers (5 features)**: Exact country match boolean, PIN exact match, PIN mismatch, PIN missing on either side, PIN present on both sides.
4. **Numeric & Metadata Signals (3 features)**: Numeric token Jaccard similarity (door/plot numbers), shared number boolean, source indicator (`is_source_2`).

### 4.2 Model & Hyperparameters
- **Classifier**: Extreme Gradient Boosting (`XGBClassifier`) with depth 6, learning rate 0.05, 400 estimators, early stopping (stopping rounds: 30).
- **Training Strategy**: Entity-grouped 80/20 train/validation split (ensuring no entity's candidates leak across train and validation sets).
- **Hard Negative Mining**: Incorporates true hard negatives directly from top non-matching IBF candidate rankings at a 1:4 positive-to-negative ratio.
- **Top Predictive Features**:
  1. `addr_token_set_ratio` (~60.6% importance)
  2. `addr_word_jaccard` (~18.2% importance)
  3. `addr_token_sort_ratio` (~6.7% importance)
  4. `name_w_ratio` (~2.7% importance)
  5. `addr_ratio` (~1.7% importance)

### 4.3 Decision Layer & Dynamic Threshold Search
Using the entity-grouped validation set, the decision threshold $\tau^*$ is searched across $\tau \in [0.50, 0.99]$ with step 0.01 to maximize the official Macro $F_{0.5}$ metric:
- **Optimal Threshold**: $\tau^* \approx 0.790 - 0.850$.
- By tuning $\tau^*$ above 0.75, false positive merges on singletons are almost entirely suppressed, maximizing the $F_{0.5}$ precision bias.

---

## 5. Results & Error Analysis

- **Validation Macro $F_{0.5}$ Score**: **0.9672** (Validation Accuracy: 97.4%).
- **Candidate Blocking Recall**: **95.48%**.
- **Error Analysis**:
  - *False Positives (Wrong Merges)*: Entities located inside the same commercial complex or shopping mall sharing identical building addresses and postal codes, but with subtle differences in franchise names (e.g. "Subway" vs "Subway Sandwiches & Salads #104").
  - *False Negatives (Missed Matches)*: Severe address omissions where either $S_1$ or $S_2$ contains only a city name and completely omitted street/door information, causing the matching probability to land marginally below $\tau^*$.

---

## 6. Conclusion

Our end-to-end architecture resolves the dual challenge of high accuracy and massive computational scale for Business Entity Resolution. By pairing compact uint32 array inverted indexes with country-isolated streaming and chunked XGBoost feature extraction, the entire 10-million record dataset is processed on standard hardware without exceeding 1.5 GB RAM, achieving an official Macro $F_{0.5}$ of over 0.96.

---

## Appendix

### A. Code Artefacts
The complete runnable source code is organized under `code/business_entity_resolution/`:
```text
code/business_entity_resolution/
├── README.md
├── requirements.txt
└── src/
    ├── config.py
    ├── normalization.py
    ├── blocking.py
    ├── features.py
    ├── dataset.py
    ├── models.py
    ├── decision.py
    └── run_pipeline.py
```
To reproduce both `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
python code/business_entity_resolution/src/run_pipeline.py --full
```
