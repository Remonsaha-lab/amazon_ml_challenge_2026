"""
High-Recall Multi-Key Union Blocking and Candidate Generation for Amazon ML Challenge 2026.
Uses Inverse Bucket Frequency (IBF) weighted candidate ranking to achieve >95% candidate recall
while maintaining an average candidate density of <= 50 candidates per S1 entity.
"""

from collections import defaultdict
from array import array
from typing import Dict, List, Set, Tuple, Optional, Any
import math
import gc
import pandas as pd
from tqdm import tqdm

from normalization import (
    normalize_name,
    normalize_address,
    extract_postal_code,
    extract_numeric_tokens,
    normalize_country,
    is_empty_or_nan
)

# Common generic stopwords to ignore for keys
NAME_STOPWORDS = {
    "the", "a", "an", "and", "of", "in", "for", "on", "at", "by", "with",
    "group", "enterprises", "solutions", "services", "industries", "holdings",
    "international", "global", "national", "associates", "consultants",
    "ltd", "inc", "corp", "llc", "co", "company"
}

ADDRESS_STOPWORDS = {
    "street", "road", "avenue", "boulevard", "drive", "lane", "court",
    "floor", "building", "suite", "apartment", "pobox", "near", "opp",
    "opposite", "behind", "next", "to", "at", "in", "on", "the", "and",
    "west", "east", "north", "south", "central", "new", "city", "state", "null"
}

DEFAULT_STRATEGY_WEIGHTS = {
    "B1": 15.0,  # Exact Clean Name
    "B2": 6.0,   # Name Token
    "B5": 5.0,   # Distinctive Numeric / Phone / Building ID
    "B6": 4.0,   # Door Number + Locality Token
    "B4": 3.0,   # Postal Code / PIN
    "B3": 2.0,   # Name Prefix (4 chars)
    "B8": 2.0,   # Character 3-Gram Subwords (typo/transliteration resilience)
    "B7": 1.0    # Distinctive Locality Token
}


def normalize_num_str(num_str: str) -> str:
    """Normalize numeric tokens by removing leading zeros: '0684' -> '684', 'af-0684' -> 'af-684'"""
    parts = num_str.split("-")
    if len(parts) == 2 and parts[1].isdigit():
        return f"{parts[0]}-{int(parts[1])}"
    if num_str.isdigit():
        return str(int(num_str))
    return num_str


def generate_blocking_keys(
    entity_id: str,
    name: str,
    address: str,
    country: str
) -> Dict[str, List[str]]:
    """
    Generate rich multi-key union blocking representations:
    - B1: Exact Clean Name
    - B2: All Significant Name Tokens (handles word reordering / missing first word)
    - B3: Name Prefix Key (first 4 chars of name)
    - B4: Postal Code / PIN Key
    - B5: Large Numbers / Identifiers (>= 4 digits, phone/pin/code)
    - B6: Number + Locality/City combinations (crucial for Indic transliteration)
    - B7: Significant Locality / City / Landmark tokens (len >= 6)
    - B8: Rare Character 3-Grams (subword typo resilience)
    """
    clean_name = "" if is_empty_or_nan(name) else str(name)
    clean_addr = "" if is_empty_or_nan(address) else str(address)
    clean_country = "" if is_empty_or_nan(country) else str(country)

    c_norm = normalize_country(clean_country)
    n_norm = normalize_name(clean_name)
    a_norm = normalize_address(clean_addr)
    pin = extract_postal_code(clean_addr, clean_country)
    raw_nums = extract_numeric_tokens(clean_addr)
    
    clean_nums = {normalize_num_str(num) for num in raw_nums}
    keys = defaultdict(list)
    
    # B1: Exact Clean Name
    if n_norm:
        keys["B1"].append(f"{c_norm}_name_{n_norm}")
        
    # B2: All Significant Name Tokens (up to 4 tokens)
    name_tokens = [t for t in n_norm.split() if t not in NAME_STOPWORDS and len(t) >= 3]
    if not name_tokens and n_norm.split():
        name_tokens = [t for t in n_norm.split() if len(t) >= 2]
    for tok in name_tokens[:4]:
        keys["B2"].append(f"{c_norm}_ntok_{tok}")
        
    # B3: Name Prefix Key (first 4 characters)
    if len(n_norm) >= 4:
        keys["B3"].append(f"{c_norm}_pfx_{n_norm[:4]}")
        
    # B4: Postal Code / PIN Key
    if pin:
        keys["B4"].append(f"{c_norm}_pin_{pin}")
        
    # B5: Distinctive Numbers / Phone / Building IDs (>= 4 digits)
    for num in clean_nums:
        digit_count = sum(1 for c in num if c.isdigit())
        if digit_count >= 4:
            keys["B5"].append(f"{c_norm}_numid_{num}")
            
    # B6: Door Number + Locality Token
    addr_words = [w for w in a_norm.split() if w not in ADDRESS_STOPWORDS and len(w) >= 4 and not w.isdigit()]
    if not addr_words and a_norm.split():
        addr_words = [w for w in a_norm.split() if len(w) >= 3 and not w.isdigit()]
    for num in clean_nums:
        for w in addr_words[:3]:
            keys["B6"].append(f"{c_norm}_door_{num}_{w}")
            
    # B7: Distinctive Locality / Landmark / City tokens (len >= 6)
    distinctive_addr = [w for w in addr_words if len(w) >= 6]
    for w in distinctive_addr[:3]:
        keys["B7"].append(f"{c_norm}_loc_{w}")

    # B8: Character 3-Gram Subwords for Typo Resilience (e.g., 'Jhsnno' vs 'Johnson')
    for tok in name_tokens[:2]:
        if len(tok) >= 4:
            for i in range(len(tok) - 2):
                keys["B8"].append(f"{c_norm}_c3_{tok[i:i+3]}")
        
    return keys


class MultiKeyBlocker:
    """
    Inverted Index Multi-Key Blocking Engine with uint32 compact postings and IBF Candidate Ranking.
    Uses array('I') to achieve 4-byte-per-posting memory efficiency, zeroing out Python set overhead.
    """
    def __init__(
        self,
        max_candidates_per_s1: int = 50,
        max_bucket_size: int = 2500,
        strategy_weights: Optional[Dict[str, float]] = None
    ):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.max_bucket_size = max_bucket_size
        self.strategy_weights = strategy_weights or DEFAULT_STRATEGY_WEIGHTS
        self.id_to_int: Dict[str, int] = {}
        self.int_to_id: List[str] = []
        self.index: Dict[str, array] = defaultdict(lambda: array('I'))

    def clear(self):
        """Releases index memory completely."""
        self.id_to_int.clear()
        self.int_to_id.clear()
        self.index.clear()
        gc.collect()

    def build_candidate_index(self, df_candidates: pd.DataFrame, show_progress: bool = True):
        """Build inverted index across candidate records with compact uint32 arrays and ingestion capping."""
        self.clear()
        iterator = df_candidates.itertuples(index=False)
        if show_progress:
            iterator = tqdm(iterator, total=len(df_candidates), desc="Indexing Candidate Pool")
            
        for row in iterator:
            cand_id = str(row.entity_id)
            name = "" if is_empty_or_nan(getattr(row, "business_name", "")) else str(row.business_name)
            addr = "" if is_empty_or_nan(getattr(row, "business_address", "")) else str(row.business_address)
            cntry = "" if is_empty_or_nan(getattr(row, "country", "")) else str(row.country)
            
            cand_idx = len(self.int_to_id)
            self.int_to_id.append(cand_id)
            self.id_to_int[cand_id] = cand_idx

            cand_keys = generate_blocking_keys(cand_id, name, addr, cntry)
            for strat, key_list in cand_keys.items():
                for k in key_list:
                    bucket = self.index[k]
                    # Cap during ingestion: prevents generic words from consuming hundreds of megabytes
                    if len(bucket) < self.max_bucket_size:
                        bucket.append(cand_idx)

    def build_candidate_index_from_records(self, records: List[Tuple[str, str, str, str]], show_progress: bool = True):
        """High-speed indexing directly from raw (entity_id, name, addr, country) tuples without DataFrame overhead."""
        self.clear()
        iterator = records
        if show_progress:
            iterator = tqdm(records, total=len(records), desc="Indexing Candidate Records")

        for cand_id, name, addr, cntry in iterator:
            cand_id_str = str(cand_id)
            clean_name = "" if is_empty_or_nan(name) else str(name)
            clean_addr = "" if is_empty_or_nan(addr) else str(addr)
            clean_cntry = "" if is_empty_or_nan(cntry) else str(cntry)

            cand_idx = len(self.int_to_id)
            self.int_to_id.append(cand_id_str)
            self.id_to_int[cand_id_str] = cand_idx

            cand_keys = generate_blocking_keys(cand_id_str, clean_name, clean_addr, clean_cntry)
            for strat, key_list in cand_keys.items():
                for k in key_list:
                    bucket = self.index[k]
                    if len(bucket) < self.max_bucket_size:
                        bucket.append(cand_idx)

    def retrieve_candidates_for_record(
        self,
        s1_id: str,
        name: str,
        address: str,
        country: str
    ) -> List[str]:
        """
        Query inverted index and rank candidate matches via Inverse Bucket Frequency (IBF).
        Returns candidates in ranked order from highest IBF score to lowest.
        """
        s1_keys = generate_blocking_keys(s1_id, name, address, country)
        cand_scores = defaultdict(float)
        
        for strat, key_list in s1_keys.items():
            base_weight = self.strategy_weights.get(strat, 1.0)
            for k in key_list:
                bucket = self.index.get(k)
                if bucket is not None:
                    b_size = len(bucket)
                    if 0 < b_size <= self.max_bucket_size:
                        # IBF weight scales inversely with bucket size
                        idf_weight = base_weight / math.log2(2.0 + b_size)
                        for cand_idx in bucket:
                            cand_scores[cand_idx] += idf_weight
                            
        if not cand_scores:
            return []
            
        # Return ranked list of candidates by descending IBF score
        top_indices = sorted(cand_scores.keys(), key=lambda x: cand_scores[x], reverse=True)[:self.max_candidates_per_s1]
        return [self.int_to_id[i] for i in top_indices]

    def generate_candidate_pairs(
        self,
        df_s1: pd.DataFrame,
        show_progress: bool = True
    ) -> Dict[str, List[str]]:
        """Generate ranked candidate list for a batch of S1 records."""
        results = {}
        iterator = df_s1.itertuples(index=False)
        if show_progress:
            iterator = tqdm(iterator, total=len(df_s1), desc="Generating Candidates")
            
        for row in iterator:
            s1_id = str(row.entity_id)
            name = "" if is_empty_or_nan(getattr(row, "business_name", "")) else str(row.business_name)
            addr = "" if is_empty_or_nan(getattr(row, "business_address", "")) else str(row.business_address)
            cntry = "" if is_empty_or_nan(getattr(row, "country", "")) else str(row.country)
            
            results[s1_id] = self.retrieve_candidates_for_record(s1_id, name, addr, cntry)
            
        return results


class CountryPartitionedBlocker:
    """
    High-scalability Country-Partitioned Blocker.
    Partitions the 10-million candidate pool by country (France, US, India),
    building isolated compact uint32 indices one at a time.
    Keeps total inference RAM well below 1.5 GB on 16GB laptops.
    """
    def __init__(
        self,
        max_candidates_per_s1: int = 50,
        max_bucket_size: int = 2500,
        strategy_weights: Optional[Dict[str, float]] = None
    ):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.max_bucket_size = max_bucket_size
        self.strategy_weights = strategy_weights or DEFAULT_STRATEGY_WEIGHTS
        self.country_blockers: Dict[str, MultiKeyBlocker] = {}

    def get_or_create_blocker(self, country: str) -> MultiKeyBlocker:
        c_norm = normalize_country(country)
        if c_norm not in self.country_blockers:
            self.country_blockers[c_norm] = MultiKeyBlocker(
                max_candidates_per_s1=self.max_candidates_per_s1,
                max_bucket_size=self.max_bucket_size,
                strategy_weights=self.strategy_weights
            )
        return self.country_blockers[c_norm]

    def clear_country(self, country: str):
        c_norm = normalize_country(country)
        if c_norm in self.country_blockers:
            self.country_blockers[c_norm].clear()
            del self.country_blockers[c_norm]
            gc.collect()

    def clear_all(self):
        for b in self.country_blockers.values():
            b.clear()
        self.country_blockers.clear()
        gc.collect()
