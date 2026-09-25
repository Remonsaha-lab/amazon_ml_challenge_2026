import sys
import os
from pathlib import Path

# Ensure UTF-8 stdout on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project root and module path to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "code" / "business_entity_resolution" / "src"))


from normalization import (
    strip_accents,
    normalize_name,
    normalize_address,
    extract_postal_code,
    extract_numeric_tokens,
    normalize_country
)

print("=" * 80)
print("BENCHMARKING & VALIDATING ROBUST NORMALIZATION ENGINE")
print("=" * 80)

# Test Cases: Real-world patterns from Ground Truth Forensics
test_cases = [
    {
        "desc": "US Name with Accents & Typos",
        "raw_name": "Payne Énterprises",
        "raw_addr": "3315 Fremont Street, Peoria, IL",
        "country": "US"
    },
    {
        "desc": "US URL Domain & Street Variations",
        "raw_name": "maurewilliamscolombier.com",
        "raw_addr": "85 Wayne Avenue, Ticonderoga, NY 12883",
        "country": "US"
    },
    {
        "desc": "India Hindi / Devanagari Script & Landmark",
        "raw_name": "एसएस फूड प्राइवेट लिमिटेड",
        "raw_addr": "Af-684, Nandgram Near Mother India Public School, Ghaziabad, UP 201003",
        "country": "India"
    },
    {
        "desc": "India Tamil Script & Alphanumeric Door",
        "raw_name": "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
        "raw_addr": "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu 600004",
        "country": "India"
    },
    {
        "desc": "French Legal Entity & Address Format",
        "raw_name": "Boulangerie Patisserie SARL",
        "raw_addr": "15 Rue de la Paix, 75002 Paris",
        "country": "France"
    },
    {
        "desc": "Shop Number with Hyphen & Sub-block",
        "raw_name": "Hotel Enterprises Limited",
        "raw_addr": "WZ-187C Shop No.13, 14 Kh. Vikaspuri, West Delhi, DL 110018",
        "country": "India"
    }
]

for idx, tc in enumerate(test_cases, 1):
    c_norm = normalize_country(tc["country"])
    n_norm = normalize_name(tc["raw_name"])
    a_norm = normalize_address(tc["raw_addr"])
    pin = extract_postal_code(tc["raw_addr"], tc["country"])
    nums = extract_numeric_tokens(tc["raw_addr"])
    
    print(f"\n[Case {idx}] {tc['desc']} ({tc['country']} -> {c_norm})")
    print(f"  Raw Name   : {tc['raw_name']}")
    print(f"  Norm Name  : {n_norm}")
    print(f"  Raw Addr   : {tc['raw_addr']}")
    print(f"  Norm Addr  : {a_norm}")
    print(f"  Postal Code: {pin}")
    print(f"  Num Tokens : {sorted(nums)}")

# Assertions for verification gate
assert normalize_name("Payne Énterprises") == "payne enterprises"
assert normalize_name("TechNova Solutions Pvt Ltd") == "technova solutions ltd"
assert normalize_name("TechNova Solutions Private Limited") == "technova solutions ltd"
assert normalize_name("maurewilliamscolombier.com") == "maurewilliamscolombier"
assert normalize_name("एसएस फूड प्राइवेट लिमिटेड") == "एसएस फूड ltd"
assert normalize_name("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி") == "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் llc"
assert normalize_country("united states") == "US"
assert normalize_country("france") == "France"
assert extract_postal_code("New York, NY 10001", "US") == "10001"
assert extract_postal_code("Chennai, Tamil Nadu 600004", "India") == "600004"
assert extract_postal_code("75002 Paris, France", "France") == "75002"
assert "wz-187c" in extract_numeric_tokens("WZ-187C Shop No.13")
assert "af-684" in extract_numeric_tokens("Af-684, Nandgram")

print("\n" + "=" * 80)
print("ALL NORMALIZATION ASSERTIONS PASSED! (Verification Gate 02 Complete)")
print("=" * 80)
