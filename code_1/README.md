# Code 1: alternative entity-resolution experiments

## Recommended alternative: grouped learning to rank

`rank_pipeline.py` trains a LightGBM LambdaRank model. Candidate pairs for each source-1 entity form a query group, and the ranker learns to place true matches above the hard negatives in that group. This changes the original pairwise binary-classification objective while retaining the repository's normalization, candidate blocker and pair features. It uses up to 100 candidates per entity by default, rather than 50.

Run from the repository root:

```bash
python code_1/rank_pipeline.py
```

To train and evaluate without generating test predictions, add `--train-only`. The pipeline tunes a score cutoff on the held-out source-1 validation groups and prints macro F0.5, mean precision/recall, and exact-set accuracy. It then writes the model and cutoff under `output/` and, unless `--train-only` is set, generates the competition TSVs.

It needs the dependencies in the repository's `requirement.txt`. Data is read through the original project's `config.py`, which resolves the root `dataset/train/` and `dataset/test/` folders.

## Deterministic comparison baseline

`strict_match.py` is a standard-library-only rules matcher that combines exact-name and postal-code blocking. It is retained for comparison. It is not the recommended route when maximizing the official score, because its precision-first rules can sacrifice recall.

The attached notebook's reported `0.9678` is validation macro F0.5, the competition metric, not ordinary accuracy. LambdaRank is a different method intended to improve within-entity ordering and recover more true matches. It may or may not improve the score; use its held-out validation report to decide.
