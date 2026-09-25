import pandas as pd
import os
import sys

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TRAIN_DIR = os.path.join("student_resource", "dataset", "train")

df_gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", nrows=50000)
df_gt = df_gt[df_gt["matched_entity_ids"].notna() & (df_gt["matched_entity_ids"] != "")].head(10)

sample_s1_ids = set(df_gt["source1_entity_id"])
all_target_ids = set()
for m in df_gt["matched_entity_ids"]:
    for x in m.split(","):
        all_target_ids.add(x.strip())

# Load S1 records
s1_rows = []
with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    header = next(f).strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in sample_s1_ids:
            s1_rows.append(parts)
            if len(s1_rows) == len(sample_s1_ids):
                break
df_s1_sample = pd.DataFrame(s1_rows, columns=header).set_index("entity_id")

# Load S2 / S3 records
s2_rows = []
with open(os.path.join(TRAIN_DIR, "train_source2.tsv"), "r", encoding="utf-8") as f:
    header_s2 = next(f).strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in all_target_ids:
            s2_rows.append(parts)

s3_rows = []
with open(os.path.join(TRAIN_DIR, "train_source3.tsv"), "r", encoding="utf-8") as f:
    header_s3 = next(f).strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in all_target_ids:
            s3_rows.append(parts)

target_dict = {}
for r in s2_rows + s3_rows:
    if len(r) >= 4:
        target_dict[r[0]] = {"name": r[1], "address": r[2], "country": r[3]}

print("\n" + "="*90)
print("REAL GROUND TRUTH MATCH INSPECTION (10 EXAMPLES)")
print("="*90)

for idx, row in df_gt.iterrows():
    s1_id = row["source1_entity_id"]
    if s1_id not in df_s1_sample.index:
        continue
    s1_data = df_s1_sample.loc[s1_id]
    m_list = [x.strip() for x in row["matched_entity_ids"].split(",")]
    
    print(f"\n[S1 Reference] {s1_id} | Country: {s1_data['country']}")
    print(f"   Name   : {s1_data['business_name']}")
    print(f"   Address: {s1_data['business_address']}")
    print(f"   Matches ({len(m_list)}):")
    for mid in m_list:
        if mid in target_dict:
            td = target_dict[mid]
            print(f"     -> [{mid}] ({td['country']})")
            print(f"        Name   : {td['name']}")
            print(f"        Address: {td['address']}")
        else:
            print(f"     -> [{mid}] (not in first chunk)")
