# -*- coding: utf-8 -*-

# Brahmic offset relative to script base (0x00 to 0x7F)
# 0x0900: Devanagari, 0x0980: Bengali, 0x0A00: Gurmukhi, 0x0A80: Gujarati
# 0x0B00: Oriya, 0x0B80: Tamil, 0x0C00: Telugu, 0x0C80: Kannada, 0x0D00: Malayalam

BRAHMIC_OFFSETS = {
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'ng',
    0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j', 0x1D: 'jh', 0x1E: 'ny',
    0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n',
    0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n',
    0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm',
    0x2F: 'y', 0x30: 'r', 0x32: 'l', 0x33: 'l', 0x35: 'v',
    0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
    0x05: 'a', 0x06: 'aa', 0x07: 'i', 0x08: 'ee', 0x09: 'u', 0x0A: 'oo',
    0x0B: 'ri', 0x0F: 'e', 0x10: 'ai', 0x13: 'o', 0x14: 'au',
    0x3E: 'a', 0x3F: 'i', 0x40: 'ee', 0x41: 'u', 0x42: 'oo',
    0x43: 'ri', 0x47: 'e', 0x48: 'ai', 0x4B: 'o', 0x4C: 'au',
    0x4D: '', 0x02: 'n', 0x01: 'n', 0x03: 'h'
}

def transliterate_brahmic(text: str) -> str:
    res = []
    for ch in text:
        cp = ord(ch)
        # Check if in Brahmic script range 0x0900 to 0x0D7F
        if 0x0900 <= cp <= 0x0D7F:
            offset = cp % 0x80
            res.append(BRAHMIC_OFFSETS.get(offset, ''))
        else:
            res.append(ch)
    return "".join(res)

if __name__ == "__main__":
    t_hindi = "राम मां लॉजिस्टिक्स प्राइवेट लिमिटेड"
    t_bengali = "ড্রিম কনস্ট্রাকশন প্রাইভেট লিমিটেড"
    t_gujarati = "ફર્સ્ટ ઇન્ડસ્ટ્રીઝ પ્રા. લિ."
    t_kannada = "ಗೋಲ್ಡ್ ಪ್ರಾಜೆಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"
    
    print("Hindi:   ", transliterate_brahmic(t_hindi))
    print("Bengali: ", transliterate_brahmic(t_bengali))
    print("Gujarati:", transliterate_brahmic(t_gujarati))
    print("Kannada: ", transliterate_brahmic(t_kannada))
