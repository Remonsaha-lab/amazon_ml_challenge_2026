"""Conservative, deterministic entity matcher for the Amazon ML challenge.

Candidate generation is exact normalized name within country. A pair is accepted
only when the normalized addresses match, or when the postal code and address
token overlap provide an independent confirmation. SQLite keeps candidate records
on disk instead of requiring a large in-memory index.
"""

from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from normalization import (
    extract_numeric_tokens,
    extract_postal_code,
    is_empty_or_nan,
    normalize_address,
    normalize_country,
    normalize_name,
)


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" if (ROOT / "dataset").exists() else ROOT / "student_resource" / "dataset"
OUT = ROOT / "output"


def rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"entity_id", "business_name", "business_address", "country"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            yield (
                (row.get("entity_id") or "").strip(),
                "" if is_empty_or_nan(row.get("business_name")) else row["business_name"].strip(),
                "" if is_empty_or_nan(row.get("business_address")) else row["business_address"].strip(),
                "" if is_empty_or_nan(row.get("country")) else row["country"].strip(),
            )


def build_index(db: sqlite3.Connection, source_paths: Iterable[Path]) -> None:
    db.execute("DROP TABLE IF EXISTS candidates")
    db.execute("CREATE TABLE candidates (id TEXT PRIMARY KEY, country TEXT, name TEXT, address TEXT, postal TEXT, numbers TEXT)")
    for path in source_paths:
        batch = []
        for eid, name, address, country in rows(path):
            n_name = normalize_name(name)
            if not eid or not n_name:
                continue
            n_address = normalize_address(address)
            postal = extract_postal_code(address, country) or ""
            numbers = "\x1f".join(sorted(extract_numeric_tokens(address)))
            batch.append((eid, normalize_country(country), n_name, n_address, postal, numbers))
            if len(batch) >= 10000:
                db.executemany("INSERT OR REPLACE INTO candidates VALUES (?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            db.executemany("INSERT OR REPLACE INTO candidates VALUES (?, ?, ?, ?, ?, ?)", batch)
        db.commit()
    db.execute("CREATE INDEX candidate_name_country ON candidates(name, country)")
    db.execute("CREATE INDEX candidate_postal_country ON candidates(postal, country)")
    db.commit()


def tokens(s: str) -> Set[str]:
    return set(s.split()) if s else set()


def similarity(a: str, b: str) -> float:
    """Conservative stdlib string similarity after canonical normalization."""
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


def predict_one(db: sqlite3.Connection, record: Tuple[str, str, str, str]):
    sid, raw_name, raw_address, raw_country = record
    name = normalize_name(raw_name)
    address = normalize_address(raw_address)
    country = normalize_country(raw_country)
    postal = extract_postal_code(raw_address, raw_country) or ""
    nums = extract_numeric_tokens(raw_address)
    # Union two compact blocks: exact canonical name, and exact postal code.
    # The postal block recovers typos/name variants; name similarity prunes
    # unrelated businesses that share a busy postal code.
    pool = {}
    if name:
        for row in db.execute(
            "SELECT id, name, address, postal, numbers FROM candidates WHERE name=? AND country=?",
            (name, country),
        ):
            pool[row[0]] = row[1:]
    if postal:
        for row in db.execute(
            "SELECT id, name, address, postal, numbers FROM candidates WHERE postal=? AND country=?",
            (postal, country),
        ):
            cid, cand_name, cand_address, cand_postal, raw_nums = row
            name_sim = similarity(name, cand_name)
            shared_name_tokens = tokens(name) & tokens(cand_name)
            if name_sim >= 0.72 or shared_name_tokens:
                pool.setdefault(cid, (cand_name, cand_address, cand_postal, raw_nums))

    candidates = []
    matches = []
    for cid, (cand_name, cand_address, cand_postal, raw_nums) in sorted(pool.items()):
        name_sim = similarity(name, cand_name)
        cand_tokens = tokens(cand_address)
        addr_tokens = tokens(address)
        token_overlap = len(addr_tokens & cand_tokens) / max(len(addr_tokens | cand_tokens), 1)
        addr_sim = max(token_overlap, similarity(address, cand_address))
        cand_nums = set(raw_nums.split("\x1f")) if raw_nums else set()
        exact_name = bool(name and name == cand_name)
        exact_address = bool(address and cand_address and address == cand_address)
        same_postal = bool(postal and cand_postal and postal == cand_postal)
        numbers_compatible = not (nums and cand_nums and nums.isdisjoint(cand_nums))
        candidates.append(cid)

        # Strong direct evidence, or a typo-tolerant match requiring independent
        # address evidence. Postal-code conflicts and building-number conflicts
        # block the fuzzy route.
        direct = exact_name and (exact_address or (same_postal and token_overlap >= 0.5))
        fuzzy = (
            name_sim >= 0.94
            and addr_sim >= 0.90
            and (not postal or not cand_postal or same_postal)
            and numbers_compatible
        )
        postal_supported = (
            same_postal
            and name_sim >= 0.88
            and addr_sim >= 0.82
            and numbers_compatible
        )
        if direct or fuzzy or postal_supported:
            matches.append(cid)
    return candidates, matches


def load_truth(path: Path) -> Dict[str, Set[str]]:
    truth = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = (row.get("source1_entity_id") or "").strip()
            raw = (row.get("matched_entity_ids") or "").strip()
            truth[sid] = {x.strip() for x in raw.split(",") if x.strip()}
    return truth


def score_validation(db: sqlite3.Connection, s1_path: Path, truth: Dict[str, Set[str]], report_path: Path) -> None:
    n_entities = n_exact = n_pred = n_true = n_tp = n_predicted_entities = 0
    macro_f05 = 0.0
    with report_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "predicted_entity_ids"))
        for rec in rows(s1_path):
            sid = rec[0]
            # Stable, entity-level holdout; no random split dependency.
            if int(hashlib.sha1(sid.encode("utf-8")).hexdigest()[:8], 16) % 5 != 0:
                continue
            _, pred_list = predict_one(db, rec)
            pred, gold = set(pred_list), truth.get(sid, set())
            writer.writerow((sid, ",".join(sorted(pred))))
            n_entities += 1
            n_pred += len(pred)
            n_true += len(gold)
            n_tp += len(pred & gold)
            n_exact += pred == gold
            n_predicted_entities += bool(pred)
            if not gold:
                entity_f05 = 1.0 if not pred else 0.0
            elif not pred or not (pred & gold):
                entity_f05 = 0.0
            else:
                p = len(pred & gold) / len(pred)
                r = len(pred & gold) / len(gold)
                entity_f05 = 1.25 * p * r / (0.25 * p + r)
            macro_f05 += entity_f05
    precision = n_tp / n_pred if n_pred else 0.0
    report = {
        "validation_entities": n_entities,
        "pair_precision_micro": precision,
        "exact_set_accuracy": n_exact / n_entities if n_entities else 0.0,
        "macro_f0_5_official": macro_f05 / n_entities if n_entities else 0.0,
        "match_coverage": n_predicted_entities / n_entities if n_entities else 0.0,
        "predicted_pairs": n_pred,
        "true_pairs": n_true,
        "true_positive_pairs": n_tp,
    }
    report_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def write_submission(db: sqlite3.Connection, s1_path: Path, matching_path: Path, candidate_path: Path) -> None:
    with matching_path.open("w", encoding="utf-8", newline="") as fm, candidate_path.open("w", encoding="utf-8", newline="") as fc:
        wm = csv.writer(fm, delimiter="\t", lineterminator="\n")
        wc = csv.writer(fc, delimiter="\t", lineterminator="\n")
        wm.writerow(("source1_entity_id", "matched_entity_ids"))
        wc.writerow(("source1_entity_id", "candidate_entity_ids"))
        for rec in rows(s1_path):
            candidates, matches = predict_one(db, rec)
            wm.writerow((rec[0], ",".join(matches)))
            wc.writerow((rec[0], ",".join(candidates)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict, precision-first business entity matching")
    parser.add_argument("--dataset", type=Path, default=DATA, help="Dataset root containing train/ and test/")
    parser.add_argument("--output", type=Path, default=OUT, help="Output directory")
    parser.add_argument("--skip-validation", action="store_true", help="Skip held-out training validation")
    parser.add_argument("--validate-only", action="store_true", help="Run held-out validation without building test predictions")
    args = parser.parse_args()
    train, test = args.dataset / "train", args.dataset / "test"
    args.output.mkdir(parents=True, exist_ok=True)
    train_s1 = train / "train_source1.tsv"
    truth_path = train / "train_ground_truth.tsv"
    train_sources = [train / "train_source2.tsv", train / "train_source3.tsv"]
    test_s1 = test / "test_source1.tsv"
    test_sources = [test / "test_source2.tsv", test / "test_source3.tsv"]
    required = [train_s1, truth_path, *train_sources, test_s1, *test_sources]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit("Missing challenge data files:\n  " + "\n  ".join(missing))

    with tempfile.TemporaryDirectory(prefix="strict_match_") as temp:
        db = sqlite3.connect(str(Path(temp) / "candidate_index.sqlite"))
        if not args.skip_validation:
            print("Validating strict rules on a stable 20% held-out slice of training source 1...")
            build_index(db, train_sources)
            score_validation(db, train_s1, load_truth(truth_path), args.output / "code1_validation.tsv")
            if args.validate_only:
                db.close()
                return
        print("Building exact-name index for test candidates...")
        build_index(db, test_sources)
        write_submission(db, test_s1, args.output / "matching_results.tsv", args.output / "candidate_pairs.tsv")
        db.close()
    print(f"Wrote submission files to {args.output}")


if __name__ == "__main__":
    main()
