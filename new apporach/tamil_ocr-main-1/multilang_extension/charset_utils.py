# Multi-language character set definitions for PARSEQ extension

# Base English charset
CHARSET_ENGLISH = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# Tamil charset (existing)
CHARSET_TAMIL = "அஆஇஈஉஊஎஏஐஒஓஔகஙசஞடணதநபமயரலவழளறன"

# Hindi Devanagari charset
CHARSET_HINDI = "ँंःअआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौ्"

# Telugu charset
CHARSET_TELUGU = "అఆఇఈఉఊఋఌఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహళ"

# Kannada charset
CHARSET_KANNADA = "ಅಆಇಈಉಊಋಌಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಳ"

# Combined charsets
CHARSET_TAMIL_ENGLISH = CHARSET_ENGLISH + CHARSET_TAMIL
CHARSET_TAMIL_ENGLISH_HINDI = CHARSET_ENGLISH + CHARSET_TAMIL + CHARSET_HINDI
CHARSET_MULTILANG = CHARSET_ENGLISH + CHARSET_TAMIL + CHARSET_HINDI + CHARSET_TELUGU + CHARSET_KANNADA

# Language mapping
LANGUAGE_CHARSETS = {
    "english": CHARSET_ENGLISH,
    "tamil": CHARSET_TAMIL,
    "hindi": CHARSET_HINDI,
    "telugu": CHARSET_TELUGU,
    "kannada": CHARSET_KANNADA
}

def get_charset_for_languages(languages):
    """Get combined charset for specified languages"""
    charset = CHARSET_ENGLISH  # Always include English
    for lang in languages:
        if lang in LANGUAGE_CHARSETS and lang != "english":
            charset += LANGUAGE_CHARSETS[lang]
    return charset