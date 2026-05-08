"""
Tamil Character Mapping Module
Maps class IDs (0-155) to Tamil Unicode characters
"""

# Tamil Unicode character mapping
# Based on common Tamil characters including vowels, consonants, and compound characters
# Unicode range: U+0B80 to U+0BFF (Tamil block)

def get_char_mapping():
    """
    Returns a dictionary mapping class IDs (0-155) to Tamil characters.
    
    Tamil script includes:
    - Independent vowels (அ, ஆ, இ, etc.)
    - Consonants (க, ங, ச, etc.)
    - Dependent vowel signs (ா, ி, ீ, etc.)
    - Numerals (௦, ௧, ௨, etc.)
    """
    
    # Tamil characters in Unicode order
    tamil_chars = [
        # Independent Vowels (12 characters)
        'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ',
        
        # Consonants (18 characters)
        'க', 'ங', 'ச', 'ஞ', 'ட', 'ண', 'த', 'ந', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன',
        
        # Dependent vowel signs (11 characters)
        'ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ',
        
        # Additional signs and symbols
        '்',  # Virama (pulli)
        'ஃ',  # Aytham
        
        # Grantha consonants (used in Sanskrit loanwords)
        'ஜ', 'ஷ', 'ஸ', 'ஹ',
        
        # Tamil numerals (10 characters)
        '௦', '௧', '௨', '௩', '௪', '௫', '௬', '௭', '௮', '௯',
        
        # Common compound characters (க் + vowels)
        'கா', 'கி', 'கீ', 'கு', 'கூ', 'கெ', 'கே', 'கை', 'கொ', 'கோ', 'கௌ',
        
        # ங் + vowels
        'ஙா', 'ஙி', 'ஙீ', 'ஙு', 'ஙூ', 'ஙெ', 'ஙே', 'ஙை', 'ஙொ', 'ஙோ', 'ஙௌ',
        
        # ச் + vowels
        'சா', 'சி', 'சீ', 'சு', 'சூ', 'செ', 'சே', 'சை', 'சொ', 'சோ', 'சௌ',
        
        # த் + vowels
        'தா', 'தி', 'தீ', 'து', 'தூ', 'தெ', 'தே', 'தை', 'தொ', 'தோ', 'தௌ',
        
        # ந் + vowels
        'நா', 'நி', 'நீ', 'நு', 'நூ', 'நெ', 'நே', 'நை', 'நொ', 'நோ', 'நௌ',
        
        # ப் + vowels
        'பா', 'பி', 'பீ', 'பு', 'பூ', 'பெ', 'பே', 'பை', 'பொ', 'போ', 'பௌ',
        
        # ம் + vowels
        'மா', 'மி', 'மீ', 'மு', 'மூ', 'மெ', 'மே', 'மை', 'மொ', 'மோ', 'மௌ',
        
        # ய் + vowels
        'யா', 'யி', 'யீ', 'யு', 'யூ', 'யெ', 'யே', 'யை', 'யொ', 'யோ', 'யௌ',
        
        # ர் + vowels
        'ரா', 'ரி', 'ரீ', 'ரு', 'ரூ', 'ரெ', 'ரே', 'ரை', 'ரொ', 'ரோ', 'ரௌ',
        
        # ல் + vowels
        'லா', 'லி', 'லீ', 'லு', 'லூ', 'லெ', 'லே', 'லை', 'லொ', 'லோ',
    ]
    
    # Create mapping dictionary
    char_map = {i: char for i, char in enumerate(tamil_chars)}
    
    # Ensure we have exactly 156 mappings
    if len(char_map) < 156:
        # Fill remaining with additional compound characters if needed
        remaining = 156 - len(char_map)
        print(f"Warning: Only {len(char_map)} characters defined. Need {remaining} more.")
        # Add placeholder for missing characters
        for i in range(len(char_map), 156):
            char_map[i] = f'[{i}]'  # Placeholder
    
    return char_map


def get_reverse_mapping():
    """Returns a dictionary mapping Tamil characters to class IDs."""
    char_map = get_char_mapping()
    return {v: k for k, v in char_map.items()}


def class_id_to_char(class_id):
    """Convert class ID to Tamil character."""
    char_map = get_char_mapping()
    return char_map.get(class_id, f'[UNK_{class_id}]')


def char_to_class_id(char):
    """Convert Tamil character to class ID."""
    reverse_map = get_reverse_mapping()
    return reverse_map.get(char, -1)


def get_charset_string():
    """
    Returns a string containing all Tamil characters for the model.
    This is used by the PARSeq model for character recognition.
    """
    char_map = get_char_mapping()
    # Return characters in order
    return ''.join([char_map[i] for i in range(len(char_map))])


if __name__ == "__main__":
    # Test the mapping
    char_map = get_char_mapping()
    print(f"Total characters mapped: {len(char_map)}")
    print(f"\nFirst 20 mappings:")
    for i in range(min(20, len(char_map))):
        print(f"  {i}: {char_map[i]}")
    
    print(f"\nCharset string length: {len(get_charset_string())}")
    print(f"Sample charset: {get_charset_string()[:50]}...")
    
    # Test reverse mapping
    reverse_map = get_reverse_mapping()
    test_char = 'அ'
    class_id = char_to_class_id(test_char)
    print(f"\n'{test_char}' -> class {class_id} -> '{class_id_to_char(class_id)}'")
