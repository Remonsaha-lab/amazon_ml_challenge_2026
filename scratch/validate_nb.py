import sys
from pathlib import Path

# Load generated notebook and test executing all python cells
import json

nb_path = Path(r"c:\Users\A\Downloads\amazon_ml_challenge_2026\code_1\kaggle_business_entity_resolution.ipynb")
with nb_path.open("r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Notebook loaded successfully. Total cells: {len(nb['cells'])}")
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Total code cells: {len(code_cells)}")
