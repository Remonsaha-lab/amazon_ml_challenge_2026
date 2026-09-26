"""
Pairwise Feature Engineering Module for Amazon ML Challenge 2026.
Computes a comprehensive 32-dimensional feature vector per candidate pair using
C++ accelerated RapidFuzz string metrics, subword character n-grams, token Jaccards,
structured door/plot number alignments, and non-linear name-address interactions.
"""

from typing import Dict, List, Any, Optional, Set
import numpy as np
from rapidfuzz import fuzz, distance

from normalization import (
    normalize_name,
    normalize_address,
    extract_postal_code,
    extract_numeric_tokens,
    normalize_country
)

FEATURE_NAMES = [
    # Name features (0-8)
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_w_ratio",
    "name_jaro_winkler",
    "name_word_jaccard",
    "name_len_diff_ratio",
    "name_exact_clean",
    
    # Address features (9-15)
    "addr_ratio",
    "addr_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_word_jaccard",
    "addr_len_diff_ratio",
    "addr_is_empty_either",
    
    # Geographic & Postal Code features (16-20)
    "country_exact",
    "pin_match",
    "pin_mismatch",
    "pin_missing_either",
    "pin_both_present",
    
    # Numeric token & Source features (21-23)
    "numeric_token_jaccard",
    "has_common_number",
    "is_source_2",
    
    # Subword Character N-Gram features (24-25)
    "name_char3_jaccard",
    "name_char4_jaccard",
    
    # Structured Door / Number Agreement (26)
    "house_num_mismatch",
    
    # Non-linear Interaction & Ambiguity features (27-31)
    "name_addr_mult",
    "name_token_set_addr_mult",
    "name_exact_pin_match",
    "strong_name_conflicting_addr",
    "pin_conflict_strong_name"
]


def _get_char_ngrams(s: str, n: int) -> Set[str]:
    """Helper to extract character n-grams from a string."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    cand_id: str,
    cand_name: str,
    cand_addr: str,
    cand_country: str,
    # Pre-normalized fields can be passed for high-throughput batching
    s1_norm_name: Optional[str] = None,
    s1_norm_addr: Optional[str] = None,
    s1_pin: Optional[str] = None,
    s1_nums: Optional[set] = None,
    cand_norm_name: Optional[str] = None,
    cand_norm_addr: Optional[str] = None,
    cand_pin: Optional[str] = None,
    cand_nums: Optional[set] = None
) -> List[float]:
    """
    Extract a 32-dimensional feature vector for a single (S1, Candidate) pair.
    """
    # 1. Normalize if not pre-computed
    n1 = s1_norm_name if s1_norm_name is not None else normalize_name(s1_name)
    n2 = cand_norm_name if cand_norm_name is not None else normalize_name(cand_name)
    
    a1 = s1_norm_addr if s1_norm_addr is not None else normalize_address(s1_addr)
    a2 = cand_norm_addr if cand_norm_addr is not None else normalize_address(cand_addr)
    
    c1 = normalize_country(s1_country)
    c2 = normalize_country(cand_country)
    
    p1 = s1_pin if s1_pin is not None else extract_postal_code(s1_addr, s1_country)
    p2 = cand_pin if cand_pin is not None else extract_postal_code(cand_addr, cand_country)
    
    nums1 = s1_nums if s1_nums is not None else extract_numeric_tokens(s1_addr)
    nums2 = cand_nums if cand_nums is not None else extract_numeric_tokens(cand_addr)

    # 2. Name Similarity Features (RapidFuzz scores normalized to [0.0, 1.0])
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_partial_ratio = fuzz.partial_ratio(n1, n2) / 100.0
    name_token_sort_ratio = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_token_set_ratio = fuzz.token_set_ratio(n1, n2) / 100.0
    name_w_ratio = fuzz.WRatio(n1, n2) / 100.0
    name_jaro_winkler = distance.JaroWinkler.similarity(n1, n2)
    
    toks1, toks2 = set(n1.split()), set(n2.split())
    name_word_jaccard = len(toks1 & toks2) / max(len(toks1 | toks2), 1)
    name_len_diff = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
    name_exact = 1.0 if (n1 and n1 == n2) else 0.0

    # 3. Address Similarity Features
    addr_empty = 1.0 if (not a1 or not a2) else 0.0
    if not addr_empty:
        addr_ratio = fuzz.ratio(a1, a2) / 100.0
        addr_partial_ratio = fuzz.partial_ratio(a1, a2) / 100.0
        addr_token_sort_ratio = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_token_set_ratio = fuzz.token_set_ratio(a1, a2) / 100.0
        
        atok1, atok2 = set(a1.split()), set(a2.split())
        addr_word_jaccard = len(atok1 & atok2) / max(len(atok1 | atok2), 1)
        addr_len_diff = abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1)
    else:
        addr_ratio = 0.0
        addr_partial_ratio = 0.0
        addr_token_sort_ratio = 0.0
        addr_token_set_ratio = 0.0
        addr_word_jaccard = 0.0
        addr_len_diff = 1.0

    # 4. Geographic & Postal Code Features
    country_exact = 1.0 if (c1 and c2 and c1 == c2) else 0.0
    
    has_p1 = p1 is not None and len(p1) > 0
    has_p2 = p2 is not None and len(p2) > 0
    both_pins = 1.0 if (has_p1 and has_p2) else 0.0
    pin_missing = 1.0 if (not has_p1 or not has_p2) else 0.0
    pin_match = 1.0 if (both_pins and p1 == p2) else 0.0
    pin_mismatch = 1.0 if (both_pins and p1 != p2) else 0.0

    # 5. Numeric Overlap & Source Indicator
    num_jaccard = len(nums1 & nums2) / max(len(nums1 | nums2), 1) if (nums1 and nums2) else 0.0
    common_num = 1.0 if (nums1 and nums2 and len(nums1 & nums2) > 0) else 0.0
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0

    # 6. Character Subword N-Gram Overlap (Typo & Transliteration resilience)
    ng3_1, ng3_2 = _get_char_ngrams(n1, 3), _get_char_ngrams(n2, 3)
    name_char3_jaccard = len(ng3_1 & ng3_2) / max(len(ng3_1 | ng3_2), 1) if (ng3_1 and ng3_2) else 0.0

    ng4_1, ng4_2 = _get_char_ngrams(n1, 4), _get_char_ngrams(n2, 4)
    name_char4_jaccard = len(ng4_1 & ng4_2) / max(len(ng4_1 | ng4_2), 1) if (ng4_1 and ng4_2) else 0.0

    # 7. Structured Door / Plot Number Disagreement
    house_num_mismatch = 1.0 if (nums1 and nums2 and len(nums1 & nums2) == 0) else 0.0

    # 8. Name <-> Address Consistency & Non-linear Interaction Signals
    name_addr_mult = name_ratio * addr_ratio
    name_token_set_addr_mult = name_token_set_ratio * addr_token_set_ratio
    name_exact_pin_match = name_exact * pin_match

    # Flag for same chain / brand name at conflicting location
    strong_name_conflicting_addr = 1.0 if (name_ratio >= 0.85 and addr_ratio < 0.40 and not addr_empty) else 0.0
    # Flag for chain store with different postal code
    pin_conflict_strong_name = 1.0 if (pin_mismatch == 1.0 and name_ratio >= 0.80) else 0.0

    return [
        name_ratio,
        name_partial_ratio,
        name_token_sort_ratio,
        name_token_set_ratio,
        name_w_ratio,
        name_jaro_winkler,
        name_word_jaccard,
        name_len_diff,
        name_exact,
        addr_ratio,
        addr_partial_ratio,
        addr_token_sort_ratio,
        addr_token_set_ratio,
        addr_word_jaccard,
        addr_len_diff,
        addr_empty,
        country_exact,
        pin_match,
        pin_mismatch,
        pin_missing,
        both_pins,
        num_jaccard,
        common_num,
        is_s2,
        name_char3_jaccard,
        name_char4_jaccard,
        house_num_mismatch,
        name_addr_mult,
        name_token_set_addr_mult,
        name_exact_pin_match,
        strong_name_conflicting_addr,
        pin_conflict_strong_name
    ]
