"""
Text and geographic normalization module for Amazon ML Challenge 2026.
Features robust accent-stripping for Latin scripts, full Indic script preservation
(Devanagari, Tamil, Kannada), multilingual legal suffix canonicalization,
international address parsing, and numeric token extraction.
"""

import re
import unicodedata
from typing import Optional, Set, Any


def is_empty_or_nan(val: Any) -> bool:
    """Check if value is None, empty, NaN float, or literal 'nan'/'none'/'null' string."""
    if val is None:
        return True
    if not isinstance(val, str):
        try:
            import pandas as pd
            if pd.isna(val):
                return True
        except Exception:
            pass
    s = str(val).strip()
    return not s or s.lower() in ("nan", "none", "null")


def strip_accents(text: str) -> str:
    """
    Remove synthetic diacritics and accents (e.g., 'Énterprises' -> 'Enterprises',
    'Bóral' -> 'Boral', 'Dréxkor' -> 'Drexkor') for Latin characters while
    preserving non-Latin scripts such as Devanagari (Hindi), Tamil, and Kannada.
    """
    if is_empty_or_nan(text):
        return ""
    
    res = []
    for ch in str(text):
        # Check if character is in extended Latin range with diacritics
        if ('\u00c0' <= ch <= '\u024f') or ('\u1e00' <= ch <= '\u1eff'):
            decomp = unicodedata.normalize("NFKD", ch)
            res.append("".join(c for c in decomp if unicodedata.category(c) != "Mn"))
        else:
            res.append(ch)
    return "".join(res)


def clean_punctuation_and_symbols(text: str) -> str:
    """
    Strips punctuation (P) and symbols (S) while preserving Letters (L),
    Indic Marks/Vowels (M), and Numbers (N).
    """
    if is_empty_or_nan(text):
        return ""
    res = []
    for c in text:
        cat = unicodedata.category(c)
        if cat.startswith(('P', 'S')) and c not in ('-', '_'):
            res.append(' ')
        else:
            res.append(c)
    return "".join(res)


# Multilingual legal suffix replacement patterns
LEGAL_PATTERNS = [
    # English / Latin
    (re.compile(r"\b(pvt\.?\s*ltd\.?|private\s+limited|pte\.?\s*ltd\.?)\b", re.IGNORECASE), " ltd "),
    (re.compile(r"\b(ltd\.?|limited)\b", re.IGNORECASE), " ltd "),
    (re.compile(r"\b(inc\.?|incorporated)\b", re.IGNORECASE), " inc "),
    (re.compile(r"\b(corp\.?|corporation)\b", re.IGNORECASE), " corp "),
    (re.compile(r"\b(llc|l\.l\.c\.|llp|l\.l\.p\.)\b", re.IGNORECASE), " llc "),
    (re.compile(r"\b(sarl|s\.a\.r\.l\.|sa|s\.a\.|sas|s\.a\.s\.)\b", re.IGNORECASE), " ltd "),
    (re.compile(r"\b(co\.?|company|cie\.?)\b", re.IGNORECASE), " co "),
    # Hindi / Devanagari legal suffixes
    (re.compile(r"(प्राइवेट\s+लिमिटेड|प्रा\.\s*लि\.|प्रा\s+लि)", re.UNICODE), " ltd "),
    (re.compile(r"(लिमिटेड|लि\.)", re.UNICODE), " ltd "),
    (re.compile(r"(कार्पोरेशन|कॉरपोरेशन)", re.UNICODE), " corp "),
    # Tamil legal suffixes
    (re.compile(r"(பிரைவேட்\s+லிமிடெட்|லிமிடெட்)", re.UNICODE), " ltd "),
    (re.compile(r"(எல்எல்பி|எல்\s*எல்\s*பி)", re.UNICODE), " llc "),
]

# URL / domain pattern
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)?([a-zA-Z0-9-]+)\.(?:com|org|net|in|io|co|fr)\b", re.IGNORECASE)
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """
    Clean and canonicalize business names across English, Hindi, Tamil, and French.
    - Strips Latin accents (É -> E)
    - Extracts domain base names from URLs
    - Canonicalizes legal entity suffixes across languages
    - Preserves Indic scripts (Devanagari, Tamil, etc.)
    - Cleans punctuation and normalizes spaces
    """
    if is_empty_or_nan(name):
        return ""
    
    text = strip_accents(str(name))
    
    # Extract domain base name if string looks like a website (e.g. 'maurewilliamscolombier.com')
    text = URL_PATTERN.sub(r"\1", text)
    
    text = text.lower()
    text = text.replace("&", " and ")
    
    # Standardize legal suffixes
    for pattern, replacement in LEGAL_PATTERNS:
        text = pattern.sub(replacement, text)
        
    # Clean punctuation while safely preserving Indic marks/vowels
    text = clean_punctuation_and_symbols(text)
    text = text.replace("_", " ")
    
    # Normalize extra whitespace
    return MULTI_SPACE_PATTERN.sub(" ", text).strip()


# Precompiled address standardizations
ADDRESS_PATTERNS = [
    (re.compile(r"\b(st|st\.|saint)\b", re.IGNORECASE), "street"),
    (re.compile(r"\b(rd|rd\.)\b", re.IGNORECASE), "road"),
    (re.compile(r"\b(ave|ave\.)\b", re.IGNORECASE), "avenue"),
    (re.compile(r"\b(blvd|blvd\.)\b", re.IGNORECASE), "boulevard"),
    (re.compile(r"\b(dr|dr\.)\b", re.IGNORECASE), "drive"),
    (re.compile(r"\b(ln|ln\.)\b", re.IGNORECASE), "lane"),
    (re.compile(r"\b(ct|ct\.)\b", re.IGNORECASE), "court"),
    (re.compile(r"\b(fl|flr|floor)\b", re.IGNORECASE), "floor"),
    (re.compile(r"\b(bldg|building)\b", re.IGNORECASE), "building"),
    (re.compile(r"\b(ste|ste\.|suite)\b", re.IGNORECASE), "suite"),
    (re.compile(r"\b(apt|apt\.|apartment)\b", re.IGNORECASE), "apartment"),
    (re.compile(r"\b(rue|r\.)\b", re.IGNORECASE), "rue"),
    (re.compile(r"\b(po\s*box|p\.o\.\s*box)\b", re.IGNORECASE), "pobox"),
]

# State abbreviation mapping (US & India common)
STATE_SYNONYMS = {
    "ny": "new york",
    "ca": "california",
    "il": "illinois",
    "oh": "ohio",
    "mo": "missouri",
    "ct": "connecticut",
    "tx": "texas",
    "fl": "florida",
    "pa": "pennsylvania",
    "up": "uttar pradesh",
    "tn": "tamil nadu",
    "rj": "rajasthan",
    "ka": "karnataka",
    "dl": "delhi",
    "mh": "maharashtra",
    "wb": "west bengal",
    "gj": "gujarat",
}


def normalize_address(address: str) -> str:
    """
    Clean and canonicalize business addresses across US, India, and France.
    - Strips Latin accents
    - Standardizes street, road, avenue, floor, suite designations
    - Expands state abbreviations
    - Retains door numbers, building codes, and landmark references
    """
    if is_empty_or_nan(address):
        return ""
    
    text = strip_accents(str(address)).lower()
    
    # Clean commas, slashes, periods into whitespace
    text = re.sub(r"[,/\\.;:()#]", " ", text)
    
    # Standardize street terms
    for pattern, replacement in ADDRESS_PATTERNS:
        text = pattern.sub(replacement, text)
        
    # Standardize common state abbreviations as discrete tokens
    tokens = text.split()
    expanded_tokens = [STATE_SYNONYMS.get(t, t) for t in tokens]
    text = " ".join(expanded_tokens)
    
    # Clean punctuation while safely preserving Indic marks/vowels
    text = clean_punctuation_and_symbols(text)
    text = text.replace("_", " ")
    return MULTI_SPACE_PATTERN.sub(" ", text).strip()


# Universal postal code pattern (matches US 5-digit, India 6-digit, France 5-digit)
PIN_INDIA_PATTERN = re.compile(r"\b([1-9][0-9]{5})\b")
PIN_US_FR_PATTERN = re.compile(r"\b([0-9]{5})\b")


def extract_postal_code(address: str, country: Optional[str] = None) -> Optional[str]:
    """
    Extract postal code / PIN / Code Postal from address.
    - India: 6 digits starting with 1-9
    - US / France: 5 digits
    """
    if is_empty_or_nan(address):
        return None
    
    addr_str = str(address)
    country_clean = (str(country) if not is_empty_or_nan(country) else "").strip().lower()
    if country_clean in ("india", "in"):
        m = PIN_INDIA_PATTERN.search(addr_str)
        if m:
            return m.group(1)
            
    m_us_fr = PIN_US_FR_PATTERN.search(addr_str)
    if m_us_fr:
        return m_us_fr.group(1)
        
    # Fallback to India 6-digit if not matched above
    m_in = PIN_INDIA_PATTERN.search(addr_str)
    if m_in:
        return m_in.group(1)
        
    return None


# Numeric & alphanumeric building/plot token pattern (e.g., '1056', '3315', 'wz-187c', 'af-684', '630')
ALPHANUM_NUMBER_PATTERN = re.compile(r"\b([a-zA-Z]{1,4}-\d+[a-zA-Z]?|\d+[a-zA-Z]?|\d+)\b")


def extract_numeric_tokens(address: str) -> Set[str]:
    """
    Extract discrete house numbers, shop numbers, building numbers, and plot codes.
    Examples: 'wz-187c', 'af-684', '3315', '1056c', '630'
    """
    if is_empty_or_nan(address):
        return set()
    
    tokens = set()
    for match in ALPHANUM_NUMBER_PATTERN.finditer(str(address).lower()):
        tok = match.group(1).strip()
        # Keep if it contains at least one digit and length <= 10
        if any(c.isdigit() for c in tok) and len(tok) <= 10:
            tokens.add(tok)
    return tokens


def normalize_country(country: str) -> str:
    """
    Open-set country normalizer. Standardizes common variants without
    hardcoding closed-world exclusions.
    """
    if is_empty_or_nan(country):
        return ""
    c = str(country).strip().lower()
    if c in ("us", "usa", "united states", "united states of america"):
        return "US"
    if c in ("india", "ind", "republic of india"):
        return "India"
    if c in ("france", "fra", "republic of france"):
        return "France"
    return str(country).strip().title()
