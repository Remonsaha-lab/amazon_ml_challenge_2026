import zipfile
import os

zip_path = "6ab10eb3b23ba_student_resource.zip"
print(f"Checking zip: {zip_path}, size: {os.path.getsize(zip_path)} bytes")

with zipfile.ZipFile(zip_path, 'r') as zf:
    namelist = zf.namelist()
    print(f"Total files in zip: {len(namelist)}")
    for name in namelist[:25]:
        info = zf.getinfo(name)
        print(f"  {name} ({info.file_size:,} bytes)")
    if len(namelist) > 25:
        print(f"  ... and {len(namelist) - 25} more files")
