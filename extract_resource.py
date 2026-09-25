import zipfile
import os

zip_path = "6ab10eb3b23ba_student_resource.zip"
print("Extracting student_resource.zip...")
with zipfile.ZipFile(zip_path, 'r') as zf:
    for member in zf.infolist():
        if member.filename.startswith("__MACOSX"):
            continue
        zf.extract(member, ".")
print("Extraction complete!")
