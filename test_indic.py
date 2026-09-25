import sys
import unicodedata
import re

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

text_hindi = "एसएस फूड प्राइवेट लिमिटेड"
text_tamil = "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"

def clean_punctuation(text):
    res = []
    for c in text:
        cat = unicodedata.category(c)
        if cat.startswith(('P', 'S')) and c not in ('-', '_'):
            res.append(' ')
        else:
            res.append(c)
    return re.sub(r'\s+', ' ', ''.join(res)).strip()

print("Cleaned Hindi:", clean_punctuation(text_hindi))
print("Cleaned Tamil:", clean_punctuation(text_tamil))
